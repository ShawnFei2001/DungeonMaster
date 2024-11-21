from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc, os, time, random
from typing import Dict, List, Tuple, Optional
import re
import warnings
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from threading import Lock
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json
import redis
from enum import Enum

# ============ Configuration ============
# GPU Configuration
torch.set_num_threads(16)
for env_var in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[env_var] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Database Configuration
DB_URL = os.getenv("DB_URL", "sqlite:///game.db")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Game Configuration
MAX_PLAYERS = 4
MAX_INVENTORY = 20
DEFAULT_HP = 20
COMBAT_TIMEOUT = 60

# Model Configuration
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3b"
FINETUNED_MODEL_PATH = "gm_model_output"
MODEL_DEVICE = "cuda"
MODEL_DTYPE = "float16"
MAX_LENGTH = 256
GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "num_beams": 1,
    "early_stopping": True,
    "pad_token_id": None
}

# Add HuggingFace token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# ============ Database Models ============
Base = declarative_base()

class Player(Base):
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    level = Column(Integer, default=1)
    experience = Column(Integer, default=0)
    hp = Column(Integer, default=20)
    max_hp = Column(Integer, default=20)
    
    inventory = relationship("Inventory", back_populates="player")
    equipment = relationship("Equipment", back_populates="player")
    combat_history = relationship("CombatHistory", back_populates="player")
    save_states = relationship("GameSave", back_populates="player")

class DBItem(Base):
    __tablename__ = 'items'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    item_type = Column(String(50), nullable=False)
    damage = Column(Integer, default=0)
    defense = Column(Integer, default=0)
    spell = Column(String(100))
    description = Column(String(500))
    rarity = Column(String(20))
    level_requirement = Column(Integer, default=1)
    drop_rate = Column(Float, default=1.0)
    value = Column(Integer, default=0)

class Inventory(Base):
    __tablename__ = 'inventory'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    item_id = Column(Integer, ForeignKey('items.id'))
    quantity = Column(Integer, default=1)
    
    player = relationship("Player", back_populates="inventory")
    item = relationship("DBItem")

class Equipment(Base):
    __tablename__ = 'equipment'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    item_id = Column(Integer, ForeignKey('items.id'))
    slot = Column(String(20), nullable=False)
    
    player = relationship("Player", back_populates="equipment")
    item = relationship("DBItem")

class Monster(Base):
    __tablename__ = 'monsters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    monster_type = Column(String(50), nullable=False)
    level = Column(Integer, default=1)
    hp = Column(Integer, nullable=False)
    damage = Column(Integer, default=0)
    defense = Column(Integer, default=0)
    experience_reward = Column(Integer, default=0)
    loot_table = Column(JSON)
    abilities = Column(JSON)

class CombatHistory(Base):
    __tablename__ = 'combat_history'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    monster_id = Column(Integer, ForeignKey('monsters.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    rounds = Column(Integer, default=0)
    victory = Column(Boolean)
    damage_dealt = Column(Integer, default=0)
    damage_taken = Column(Integer, default=0)
    experience_gained = Column(Integer, default=0)
    loot_gained = Column(JSON)
    
    player = relationship("Player", back_populates="combat_history")
    monster = relationship("Monster")

class GameSave(Base):
    __tablename__ = 'game_saves'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    game_state = Column(JSON)
    
    player = relationship("Player", back_populates="save_states")

# ============ Game Models ============
@dataclass
class Item:
    name: str
    item_type: str
    damage: int = 0
    defense: int = 0
    spell: Optional[str] = None
    
    def get_spell_description(self) -> str:
        return f" [Spell: {self.spell}]" if self.spell else ""
    
    def __str__(self) -> str:
        stats = []
        if self.damage > 0:
            stats.append(f"DMG:{self.damage}")
        if self.defense > 0:
            stats.append(f"DEF:{self.defense}")
        stats_str = ", ".join(stats)
        return f"{self.name} ({stats_str}){self.get_spell_description()}"

class EntityStatus:
    def __init__(self, name: str, entity_type: str, hp: int = 20):
        self.name = name
        self.entity_type = entity_type
        self.hp = hp
        self.max_hp = hp
        self.inventory: Dict[str, Item] = {}
        self.equipment_slots = {
            "weapon": None,
            "offhand": None,
            "armor": None,
            "accessory": None
        }
        self._status_lock = Lock()
    
    def equip_item(self, item_name: str) -> str:
        with self._status_lock:
            if item_name not in self.inventory:
                return f"❌ {item_name} not found in inventory"
                
            item = self.inventory[item_name]
            slot = {
                "weapon": "weapon",
                "shield": "offhand",
                "armor": "armor",
                "accessory": "accessory"
            }.get(item.item_type)
            
            if not slot:
                return f"❌ Cannot equip {item_name}"
                
            if self.equipment_slots[slot]:
                self.equipment_slots[slot] = None
                
            self.equipment_slots[slot] = item
            return f"✅ Equipped {item_name} to {slot} slot"
    
    def get_status_report(self) -> str:
        return (f"{self.name} ({self.entity_type})\n"
                f"HP: {self.hp}/{self.max_hp} {self.get_hp_bar()}")
    
    def get_hp_bar(self) -> str:
        bar_length = 20
        fill = int((self.hp / self.max_hp) * bar_length)
        color = "🟩" if fill == bar_length else "🟨" if fill > bar_length // 2 else "🟥"
        return f"{color * fill}{'⬜' * (bar_length - fill)}"

# ============ Game Logic ============
class GameState:
    def __init__(self, db):
        self.db = db
        self.round_number = 0
        self.entities: Dict[str, EntityStatus] = {}
        self.combat_scene = None
    
    def get_entities(self, entity_type: str) -> Dict[str, EntityStatus]:
        return {name: entity for name, entity in self.entities.items()
                if entity.entity_type == entity_type}
    
    def save_state(self, player_id: int):
        state = {
            "round": self.round_number,
            "scene": self.combat_scene,
            "entities": {
                name: entity.__dict__ 
                for name, entity in self.entities.items()
            }
        }
        self.db.save_game_state(player_id, state)

class SimpleGM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            # Initialize tokenizer with proper settings for Llama
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH,
                use_auth_token=True,
                trust_remote_code=True,
                legacy=True,  # Llama tokenizer
                add_bos_token=True,
                add_eos_token=True,
                padding_side="right"  # Change to right padding
            )
            
            # Ensure special tokens are set
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=getattr(torch, MODEL_DTYPE),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_auth_token=True
            )
            
            # Load fine-tuned weights if they exist
            if os.path.exists(FINETUNED_MODEL_PATH):
                print("Loading fine-tuned weights...")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    FINETUNED_MODEL_PATH,
                    torch_dtype=getattr(torch, MODEL_DTYPE)
                )
            
            self.model.eval()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
            
        self.response_cache = {}
        self.max_cache_size = 1000
    
    def get_from_cache(self, prompt: str) -> Optional[str]:
        if self.redis_client:
            cached = self.redis_client.get(prompt)
            if cached:
                return cached.decode('utf-8')
        return self.local_cache.get(prompt)
    
    def set_in_cache(self, prompt: str, response: str):
        if self.redis_client:
            self.redis_client.setex(prompt, 3600, response)  # Cache for 1 hour
        else:
            if len(self.local_cache) >= self.max_cache_size:
                self.local_cache.clear()
            self.local_cache[prompt] = response
    
    def clean_response(self, text: str) -> str:
        """Clean up model response"""
        # Remove repetitive "Continue the story:" prompts
        text = re.sub(r'Continue the story:\s*', '', text)
        
        # Remove repetitive content
        lines = text.split('\n')
        unique_lines = []
        for line in lines:
            if line and line not in unique_lines[-2:]:  # Check against last 2 lines
                unique_lines.append(line)
        
        return '\n'.join(unique_lines).strip()
    
    def generate_response(self, prompt: str, max_length: int = None) -> str:
        try:
            # Check cache first
            cached_response = self.get_from_cache(prompt)
            if cached_response:
                return cached_response
            
            # Prepare input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length or MAX_LENGTH
            ).to(self.device)
            
            # Generate with optimized parameters
            with torch.no_grad(), autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    temperature=GENERATION_CONFIG["temperature"],
                    top_p=GENERATION_CONFIG["top_p"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    num_beams=GENERATION_CONFIG["num_beams"],
                    early_stopping=GENERATION_CONFIG["early_stopping"],
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self.clean_response(response)
            
            # Validate response isn't too repetitive
            if len(set(response.split())) < len(response.split()) / 3:
                return "The party faces an unexpected situation."
                
            # Cache the response
            self.set_in_cache(prompt, response)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "An unexpected event occurs."
    
    def cleanup(self):
        """Clean up resources"""
        self.response_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

# ============ Utility Functions ============
def create_monster(monster_type: str) -> EntityStatus:
    monster_templates = {
        "goblin": {"hp": 15, "weapon": Item("Rusty Dagger", "weapon", damage=3)},
        "orc": {"hp": 25, "weapon": Item("Battle Axe", "weapon", damage=5)},
        "troll": {"hp": 40, "weapon": Item("Club", "weapon", damage=7)},
        "dragon": {"hp": 100, "weapon": Item("Dragon Breath", "weapon", damage=15)},
    }
    
    template = monster_templates[monster_type]
    monster = EntityStatus(f"{monster_type.title()}", "monster", hp=template["hp"])
    monster.inventory[template["weapon"].name] = template["weapon"]
    monster.equip_item(template["weapon"].name)
    return monster

def get_player_action(player: EntityStatus, game_state) -> str:
    print(f"\nAvailable actions for {player.name}:")
    print("1. Attack <target>")
    print("2. Use <item>")
    print("3. Equip <item>")
    print("4. Status")
    print("5. Inventory")
    
    while True:
        action = input("\nEnter your action: ").strip()
        if action.startswith('/') or is_valid_action(action, player, game_state):
            return action
        print("Invalid action. Try again.")

def is_valid_action(action: str, player: EntityStatus, game_state) -> bool:
    action_lower = action.lower()
    
    # Check basic commands
    if action_lower in ['status', 'inventory']:
        return True
    
    # Parse attack commands
    if action_lower.startswith('attack'):
        target_name = action_lower.replace('attack', '').strip()
        return any(e.name.lower() == target_name for e in game_state.entities.values())
    
    # Parse equipment commands
    if action_lower.startswith('equip'):
        item_name = action_lower.replace('equip', '').strip()
        return item_name in player.inventory
    
    return False

# ============ Main Game Loop ============
class SceneType(Enum):
    STORY = "story"
    EVENT = "event"
    COMBAT = "combat"
    REWARD = "reward"

@dataclass
class Scene:
    type: SceneType
    description: str
    choices: Optional[List[str]] = None
    monsters: Optional[List[EntityStatus]] = None
    rewards: Optional[Dict[str, Item]] = None

class GameParty:
    def __init__(self, player_count: int):
        self.players: List[EntityStatus] = []
        self.current_scene: Optional[Scene] = None
        self.scene_history: List[Scene] = []
        self.shared_inventory: Dict[str, Item] = {}
        self.quest_progress: Dict[str, bool] = {}
        
    def add_player(self, name: str) -> EntityStatus:
        player = EntityStatus(name, "player")
        self.players.append(player)
        return player
    
    def get_living_players(self) -> List[EntityStatus]:
        return [p for p in self.players if p.hp > 0]

class GameMaster:
    def __init__(self):
        self.gm = SimpleGM()
        self.current_party: Optional[GameParty] = None
        self.story_context = ""
        
    def start_new_game(self) -> str:
        print("Welcome to the Multiplayer Adventure!")
        player_count = int(input("Enter number of players (1-4): "))
        
        if not 1 <= player_count <= 4:
            raise ValueError("Player count must be between 1 and 4")
            
        self.current_party = GameParty(player_count)
        
        # Get player names and create characters
        for i in range(player_count):
            name = input(f"Enter name for Player {i+1}: ")
            self.current_party.add_player(name)
            
        # Generate story background
        story = self.generate_story_background()
        print("\n" + "="*60)
        print("📜 Your Adventure Begins...")
        print("="*60)
        print(story)
        
        return story
    
    def generate_story_background(self) -> str:
        players = ", ".join(p.name for p in self.current_party.players)
        prompt = f"""You are a fantasy game master. Create a brief, engaging story introduction for {players}.
Include:
- A unique personal motivation
- A clear main quest
- A current situation
Keep it under 3 paragraphs and make it specific.
"""
        self.story_context = self.gm.generate_response(prompt, max_length=200)
        return self.story_context
    
    def generate_scene(self) -> Scene:
        # Weighted scene selection
        scene_weights = {
            SceneType.STORY: 0.2,
            SceneType.EVENT: 0.5,
            SceneType.COMBAT: 0.3
        }
        scene_type = random.choices(
            list(scene_weights.keys()),
            weights=list(scene_weights.values()),
            k=1
        )[0]
        
        if scene_type == SceneType.STORY:
            return self.generate_story_scene()
        elif scene_type == SceneType.EVENT:
            return self.generate_event_scene()
        else:
            return self.generate_combat_scene()
    
    def generate_story_scene(self) -> Scene:
        prompt = f"""Based on the current story:
{self.story_context}

Create a brief story development scene that advances the plot.
Focus on one specific event or revelation.
Keep it under 2 paragraphs."""

        description = self.gm.generate_response(prompt, max_length=150)
        self.story_context += "\n" + description
        return Scene(SceneType.STORY, description)
    
    def generate_event_scene(self) -> Scene:
        prompt = f"""Create a specific event scene with exactly 2 meaningful choices.
Current context: {self.story_context[-200:]}

Format:
[Description]
Choice 1: [Action 1]
Choice 2: [Action 2]

Make choices have clear potential consequences."""
        
        try:
            response = self.gm.generate_response(prompt, max_length=200)
            parts = response.split("\nChoice")
            
            description = parts[0].strip()
            choices = [
                choice.split(": ", 1)[1].strip() 
                for choice in parts[1:] 
                if ": " in choice
            ]
            
            if len(choices) < 2:
                choices = ["Proceed cautiously", "Take a risk"]
            
            return Scene(SceneType.EVENT, description, choices=choices)
            
        except Exception as e:
            print(f"Error generating event scene: {str(e)}")
            return Scene(
                SceneType.EVENT,
                "You encounter a mysterious situation.",
                choices=["Proceed cautiously", "Take a risk"]
            )
    
    def generate_combat_scene(self) -> Scene:
        # Generate appropriate monsters based on party size/level
        party_size = len(self.current_party.players)
        monster_count = random.randint(1, party_size + 1)
        
        monsters = []
        for _ in range(monster_count):
            monster_type = random.choice(["goblin", "orc", "troll", "dragon"])
            monsters.append(create_monster(monster_type))
            
        prompt = f"""Create a combat encounter description.
Party: {', '.join(p.name for p in self.current_party.players)}
Monsters: {', '.join(m.name for m in monsters)}"""
        
        description = self.gm.generate_response(prompt, max_length=150)
        return Scene(SceneType.COMBAT, description, monsters=monsters)

class CombatManager:
    def __init__(self, party: GameParty, monsters: List[EntityStatus]):
        self.party = party
        self.monsters = monsters
        self.initiative_order = []
        self.round = 0
        
    def roll_initiative(self):
        """Determine turn order for combat"""
        all_combatants = self.party.players + self.monsters
        self.initiative_order = sorted(
            all_combatants,
            key=lambda x: random.random()  # Simple random initiative for now
        )
        
    def run_combat(self) -> bool:
        """Run the combat encounter, returns True if party wins"""
        print("\n⚔️ Combat Begins!")
        self.roll_initiative()
        
        while True:
            self.round += 1
            print(f"\nRound {self.round}")
            
            for entity in self.initiative_order:
                if entity.hp <= 0:
                    continue
                    
                print(f"\n👉 {entity.name}'s turn")
                
                if entity.entity_type == "player":
                    action = get_player_action(entity, self)
                    if action.startswith('/'):
                        if action == '/quit':
                            return False
                        continue
                else:
                    # Monster AI
                    target = random.choice(self.party.get_living_players())
                    weapon = next((item for item in entity.inventory.values() if item.damage > 0), None)
                    action = f"Attack {target.name} with {weapon.name if weapon else 'claws'}"
                
                # Process action and update HP
                result, hp_changes = self.process_combat_action(entity, action)
                print(result)
                
                # Check victory/defeat conditions
                if not self.party.get_living_players():
                    return False
                if all(m.hp <= 0 for m in self.monsters):
                    return True

def main():
    game_master = None
    try:
        print("\n🎲 Welcome to the Fantasy Adventure!")
        game_master = GameMaster()
        story = game_master.start_new_game()
        
        # Main game loop
        while True:
            # Generate new scene
            scene = game_master.generate_scene()
            game_master.current_party.current_scene = scene
            
            print("\n" + "="*60)
            print(f"📜 {scene.type.value.title()} Scene")
            print("="*60)
            print(scene.description)
            
            if scene.type == SceneType.COMBAT:
                # Handle combat
                combat_manager = CombatManager(game_master.current_party, scene.monsters)
                victory = combat_manager.run_combat()
                
                if not victory:
                    print("\n💀 Game Over - The party has fallen!")
                    break
                    
            elif scene.type == SceneType.EVENT:
                # Handle event choices
                print("\nChoices:")
                for i, choice in enumerate(scene.choices, 1):
                    print(f"{i}. {choice}")
                    
                # Get choice from each player
                for player in game_master.current_party.players:
                    choice = int(input(f"\n{player.name}, enter your choice (1-{len(scene.choices)}): "))
                    # Process choice consequences
            
            # Check if party wants to continue
            if input("\nContinue to next scene? (y/n): ").lower() != 'y':
                break
                
    except Exception as e:
        print(f"\nGame error: {str(e)}")
        
    finally:
        if game_master and hasattr(game_master, 'gm'):
            game_master.gm.cleanup()
        print("Game ended.")

if __name__ == "__main__":
    main()