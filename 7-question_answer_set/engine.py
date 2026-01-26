import os
import re
import json
import shutil
import random
import unicodedata
import requests
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter

# --- CONFIGURATION ---
LLM_MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
FORCE_REBUILD = False

# --- PATHS ---
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RULES_STORAGE = DATA_DIR / "storage_rules"
CR_TXT_PATH = DATA_DIR / "magic_comprehensive_rules.txt"
CR_URL = "https://media.wizards.com/2025/downloads/MagicCompRules%2020251114.txt"

CARDS_STORAGE = DATA_DIR / "storage_cards"
ATOMIC_URL = "https://mtgjson.com/api/v5/AtomicCards.json"
ATOMIC_PATH = DATA_DIR / "AtomicCards.json"
CARDS_JSONL = DATA_DIR / "mtg_cards_clean.jsonl"
CARD_NAMES_FILE = DATA_DIR / "card_names.json"

# --- THE CASE LAW LIBRARY (STRICT AXIOM MAPPING) ---
CASE_LAW_DOCS = [
    Document(
        text="""
        *** LEGAL PRINCIPLE: LAYERS & PERSISTENCE (AXIOM 1) ***
        **The Rule**: Apply Layer logic (Rule 613) strictly.
        
        **Scenario A: The Victim (e.g., Urza's Saga vs Blood Moon)**
        Effects that change a permanent's subtype (Layer 4) wipe all abilities in the *printed* text box. 
        HOWEVER, abilities granted by *other* resolved effects (in Layer 6) are NOT removed.
        *Precedent*: **Urza's Saga** (Land) under **Blood Moon** becomes a Mountain. It loses printed abilities. 
        BUT, if it has gained an ability via a resolved Chapter trigger (Layer 6), it **RETAINS** that gained ability.

        **Scenario B: The Source (e.g., Dryad of the Ilysian Grove)**
        If a static ability *grants* types (Layer 4), and the source later loses its abilities (Layer 6), the type change **PERSISTS**.
        *Precedent*: **Merfolk Trickster** (Layer 6) removes **Dryad of the Ilysian Grove**'s abilities. The lands **REMAIN** all basic land types because Layer 4 happens before Layer 6.

        **Scenario C: Timestamps**
        If two Layer 4 effects conflict, the latest Timestamp wins.
        *Precedent*: **Purphoros** (God) attached with **Unable to Scream** (Toy). Unable to Scream enters *after* Purphoros. Purphoros becomes a Toy and loses his "Devotion" ability. He stays a Toy forever.
        """,
        metadata={"topic": "layers_blood_moon_urza_dryad"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: SAGA SACRIFICE CHECK (AXIOM 2) ***
        **The Rule**: A Saga is sacrificed (Rule 704.5s) ONLY if two conditions are met simultaneously:
        1. The number of counters >= The final chapter number.
        2. AND the Saga actually possesses chapter abilities.
        
        **The Logic Gate**: If a continuous effect (Layer 4, e.g., Blood Moon, Song of the Dryads) removes the *printed* chapter abilities, the Saga currently has 0 chapter abilities. 
        **Verdict**: Condition 2 FAILS. The Saga is **NOT** sacrificed. It survives in stasis (usually as a Mountain) and retains any abilities gained from resolved chapters.
        """,
        metadata={"topic": "sagas_sacrifice_blood_moon"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: THE STACK, LIFO & TARGETING (AXIOM 3) ***
        **1. LIFO (Last In, First Out)**: The Stack resolves in reverse.
        *Precedent (The One Ring)*: P1 activates **The One Ring**. P2 responds with **Tear Asunder**.
        **Sequence**: Tear Asunder resolves FIRST (Ring is Exiled). Ring Ability resolves LAST.
        **LKI**: Since the Ring is gone, use **Last Known Information**. Count counters *just before* it left. P1 draws cards.
        
        **2. Target First, Pay Later**: You choose Targets *when you put the ability on the stack*, NOT when it resolves. You do NOT need to be able to pay the resolution cost (Processing) to declare the target.
        *Precedent (Wasteland Strangler)*: You cast **Wasteland Strangler**. You TARGET **Phantasmal Bear**. You do NOT need a card in exile to target it.
        **Outcome**: The Bear becomes a target. Its "Sacrifice when targeted" ability triggers. The Bear dies. Then Strangler resolves (and fails to process), but the Bear is already dead.
        
        **3. The Valid Cast Check**: You must have a legal target *at the moment of casting*. 
        *Precedent*: You **CANNOT** cast **Spell Pierce** on an empty stack just to trigger Prowess.
        
        **4. The Trickster Rule**: If a source leaves play *before* its ETB trigger resolves, any "Leaves Battlefield" triggers happens *before* the "Enters Battlefield" effect can finish.
        *Precedent*: **Tidehollow Sculler** enters. Response: **Cloudshift**. Sculler leaves (triggering "Return") then returns new. "Return" resolves first (returns nothing). Then "Exile" resolves (exiles forever).

        **5. Ward**: Ward is a Triggered Ability. It uses the stack.
        *Precedent*: You can Stifle or **Consign to Memory** a Ward trigger (e.g., **Kappa Cannoneer**). If countered, the tax is not paid, and the spell resolves.
        """,
        metadata={"topic": "stack_lifo_onering_strangler"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: MATH, SEQUENCES & LKI (AXIOM 4) ***
        **1. Last Known Information (LKI)**: If an ability requires info from a source that is gone, use the state of the source **immediately before** it left.
        
        **2. Sequential Actions**: If a card uses two verbs ("Do A, *then* Do B"), these are separate events. Replacement effects apply to *each* event.
        *Precedent*: **Field-Tested Frying Pan** creates a Food, *then* a Halfling. **Peregrin Took** adds +1 Food to Event 1 AND +1 Food to Event 2. Total: 3 Foods, 1 Halfling.

        **3. Batch Counters**: "Put X counters" is ONE event.
        *Precedent*: **Clown Car** enters with 3 counters. **Omarthis** or **The Ozolith** triggers **ONCE**, not 3 times.
        
        **4. Multipliers**: "Deal X damage for *each* Y".
        *Precedent*: 2 damage for each card in exile (2 cards) = 4 damage total.
        """,
        metadata={"topic": "math_lki_sequences_counters"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: XEROX RULE, ANIMATION & CONTROL (AXIOM 5) ***
        **1. The Xerox Rule**: Copy effects (Rule 706.2) acquire ONLY the original card's *printed* text (Name, Cost, Traits).
        **Exclusions**: Copies ignore Status (tapped), Counters, Auras, and **Temporary Effects**.

        **2. The Animation Trap**: If you copy a permanent that is temporarily a creature (e.g., an activated Mutavault, Gideon, or Vehicle), the copy is the **PRINTED CARD**.
        *Precedent*: **Phantasmal Image** copying an activated **Mutavault**.
        *Verdict*: The Image becomes a **LAND** (the printed card). It is **NOT** a creature. It cannot attack or block. It does not trigger "Creature enters" effects.

        **3. Cast vs Copy**: Copying a permanent is NOT Casting. Creating a copy on the stack is NOT Casting.
        *Precedent*: **Sculpting Steel** copying **Cityscape Leveler** does NOT trigger "When you cast this spell".
        *Precedent*: Replicating **Shattering Spree** does NOT trigger Prowess for the copies.

        **4. Continuous Control Effects (The Treachery Rule)**: Control-changing effects apply **simultaneously** with the permanent entering the battlefield. There is NO moment where the opponent controls it on the battlefield.
        *Precedent*: You cast **Treachery** on opponent's **Eidolon of Blossoms**. When Treachery enters, you gain control of Eidolon *immediately*. The game then checks for triggers. Since you control Eidolon, its "Constellation" trigger fires for **YOU**.
        """,
        metadata={"topic": "copy_clone_mutavault_animation_control"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: COMBAT & DAMAGE (AXIOM 6) ***
        **1. The Ghost Block**: Once blocked, an attacker remains blocked. Removing the blocker does NOT let damage through to the player (unless Trample).
        *Precedent*: 5/5 blocked by 1/1. 1/1 sacrificed. 5/5 deals **0 damage** to player.
        
        **2. Lifelink Dependency**: Lifelink requires actual damage.
        *Precedent*: If blocked by a missing creature (Ghost Block), damage is 0. Life gained is 0.

        **3. Summoning Sickness Scope**: Sickness prevents attacking or using `{T}` abilities of the creature itself.
        *Precedent*: It does **NOT** prevent tapping for costs of *other* cards. You **CAN** tap a summoning-sick Elf for **Jaspera Sentinel**, **Convoke**, or **Earthcraft**.
        """,
        metadata={"topic": "combat_damage_ghost_block_sickness"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: ZONES, SBAs & HIDDEN INFO (AXIOM 7) ***
        **1. SBA Timing (The Blink Rule)**: State-Based Actions (Rule 704, dying from 0 toughness) are **NEVER CHECKED** in the middle of a resolving spell or ability. They are only checked when a player receives priority.
        *Precedent*: **Ephemerate** on **Master of Waves**. 
        **Sequence**: 1. Master leaves (Tokens become 0/0). 2. Master returns immediately. 3. Spell finishes. 4. SBAs check.
        **Verdict**: Since Master is back when SBAs check, the tokens are 2/1. They **SURVIVE**.
        
        **2. Face-Down Characteristics**: Face-down cards (Manifest/Morph/Cloak) have NO name and NO abilities on the battlefield.
        *Precedent*: Manifested **Nexus of Fate** or **Blightsteel Colossus** dies. 
        **Analysis**: The "Shuffle into Library" replacement effect is looked for on the *Battlefield* state. Since the face-down creature has no abilities, the effect DOES NOT exist. 
        **Verdict**: The card goes to the Graveyard and **STAYS THERE**.
        
        **3. Aura Mortality**: If a permanent changes type (Forest -> Mountain via **Harbinger**) and an Aura becomes illegal, the Aura dies (SBA).
        
        **4. Adventure & Plot**: 
        - Adventure spells go to **Exile** on resolution, not Graveyard.
        - **Plot Ownership**: If you cast opponent's card (Gonti) and Plot it (Aven Interrupter), the **OWNER** gets to cast it later.
        
        **5. Sideboard**: The Sideboard is a **Hidden Zone** outside the game. You **cannot** look at it during a game unless an effect (Wish) specifically allows it.

        **6. Static Ability Scope**: 
        - **Permanents vs. Cards**: If an ability affects 'Permanents', it refers **ONLY** to objects on the Battlefield (Rule 110.1). 
        - **Internal Zones**: If an ability affects 'Permanent cards you own that aren't on the battlefield', it refers **ONLY** to internal Game Zones (Hand, Graveyard, Library, Exile, Stack, and Command Zone). 
        - **The Sideboard Barrier**: The Sideboard is 'Outside the Game' and is **NOT** a game zone. Static abilities on the battlefield never affect cards 'Outside the Game' unless the effect specifically mentions that zone (which is extremely rare).
        *Precedent*: **Encroaching Mycosynth** makes permanents on the board and cards in your hand/graveyard artifacts, but it cannot reach into the Sideboard. **Karn, the Great Creator** can only fetch cards from the sideboard that are naturally artifacts; he cannot 'see' the Mycosynth effect on cards outside the game.
        """,
        metadata={"topic": "zones_sba_flicker_facedown_nexus_scope"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: PROTECTION & INTERACTION (AXIOM 8) ***
        **1. Hexproof/Indestructible**: 
        - Hexproof prevents **Targeting**. It does NOT stop global wipe effects (Damnation).
        - Indestructible prevents **Destroy** and **Lethal Damage**. It does NOT prevent dying from 0 Toughness (Toxic Deluge).
        
        **2. Graveyard vs Exile**: "Dies" means "Battlefield -> Graveyard".
        *Precedent*: **Swords to Plowshares** exiles. This does **NOT** trigger "When this creature dies" effects.
        
        **3. The Legend Rule**: Applies ONLY if two permanents share the **EXACT** English name.
        *Precedent*: **Cecil, Dark Knight** and **Cecil, Redeemed Paladin** are different names. They **CAN** coexist.
        
        **4. Subtype Matching**: Effects looking for "Forest" look at the type line. **Snow-Covered Forest** IS a Forest.
        """,
        metadata={"topic": "protection_indestructible_legend_dies"}
    ),
    Document(
        text="""
        *** LEGAL PRINCIPLE: COSTS & REQUIREMENTS (AXIOM 9) ***
        **1. Payment is Instant**: Sacrificing/paying life happens *during* cast. Opponents cannot respond to prevent payment.
        *Precedent*: You cannot kill a creature being sacrificed to **Deadly Dispute** to stop the spell.
        
        **2. No Forced Payment**: You are never forced to pay mana/life to attack (e.g. **Ghostly Prison**), even if you can.
        
        **3. Equip Restriction**: "Equip" targets "Creature YOU control".
        *Precedent*: You **CANNOT** redirect an opponent's Equip (e.g. **Shadowspear**) to your **Spellskite**. You do not control the Equip source.
        
        **4. Trigger Timing**: Triggers checking "When becomes target" (e.g. **Unsettled Mariner**, **Rayne**, **Spellskite**) must exist *before* the targeting event.
        *Precedent*: Flashing in **Unsettled Mariner** in response to **Fatal Push** is TOO LATE. The targeting event has already passed.
        
        **5. Processor Success**: Moving a card from Exile to GY is a "Cost". If a replacement effect (Rest in Peace) moves it back, the cost is still PAID.
        *Precedent*: **Oracle of Dust** works under **Rest in Peace**.

        **6. The Impossibility of Payment**: You cannot pay a cost unless you have the requisite resources.
        *Precedent (Bolas's Citadel + Platinum Angel)*: **Platinum Angel** prevents you from losing the game, but it does NOT grant you "infinite life" to spend. If you are at 5 life, you **CANNOT** pay 6 life to cast a spell. The payment is illegal because you do not have the resources.
        """,
        metadata={"topic": "costs_equip_redirect_mariner_processor_limits"}
    )
]

# --- SETUP ---
def load_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        load_dotenv()
        load_dotenv(Path("..") / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not found. Please set it in .env.\n")
        exit(1)

# --- HELPERS ---
def clean_unicode(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(ch for ch in s if (unicodedata.category(ch) not in {"Cf", "Cc", "Cs", "Co", "Cn"} or ch in ("\n", "\t")))

def clean_val(val):
    return str(val).strip() if val is not None else ""

# --- INGESTION LOGIC ---
def parse_rules_logic(txt_path: Path) -> List[Dict[str, Any]]:
    text = txt_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    chapter_pattern = re.compile(r"^([1-9])\.\s+(.*)$")
    section_pattern = re.compile(r"^(\d{3})\.\s+(.*)$")
    rule_pattern = re.compile(r"^(\d{3}\.\d+[a-z]?)\.?\s+(.*)$")

    nodes = []
    current_chapter_id = "1"; current_chapter_title = "Game Concepts"
    current_section_id = "100"; current_section_title = "General"
    current_rule_id = None; current_rule_lines = []
    glossary_mode = False; glossary_lines = []; rules_parsed_count = 0

    print("[LOG] Starting Rules Parsing...")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if glossary_mode: glossary_lines.append(line)
            continue
        if stripped == "Glossary":
            if rules_parsed_count > 100:
                if current_rule_id:
                    nodes.append({"rule_id": current_rule_id, "chapter_id": current_chapter_id, "chapter_title": current_chapter_title, "section_id": current_section_id, "section_title": current_section_title, "text": "\n".join(current_rule_lines).strip()})
                    current_rule_id = None
                glossary_mode = True; continue
            else: continue
        if stripped == "Credits":
            if glossary_mode: break
            continue
        if not glossary_mode:
            m_chap = chapter_pattern.match(line)
            if m_chap: current_chapter_id = m_chap.group(1); current_chapter_title = m_chap.group(2).strip(); continue
            m_sec = section_pattern.match(line)
            if m_sec: current_section_id = m_sec.group(1); current_section_title = m_sec.group(2).strip(); continue
            m_rule = rule_pattern.match(line)
            if m_rule:
                if current_rule_id:
                    nodes.append({"rule_id": current_rule_id, "chapter_id": current_chapter_id, "chapter_title": current_chapter_title, "section_id": current_section_id, "section_title": current_section_title, "text": "\n".join(current_rule_lines).strip()})
                current_rule_id = m_rule.group(1); current_rule_lines = [m_rule.group(2).strip()]; rules_parsed_count += 1
                if not current_section_id: current_section_id = current_rule_id.split('.')[0]
                if not current_chapter_id: current_chapter_id = current_section_id[0]
            elif current_rule_id: current_rule_lines.append(line)
        else: glossary_lines.append(line)
    
    if glossary_lines:
        print(f"[LOG] Processing {len(glossary_lines)} lines of glossary...")
        full_text = "\n".join(glossary_lines)
        entries = re.split(r'\n\s*\n', full_text.strip())
        for entry in entries:
            parts = entry.strip().split('\n', 1)
            if not parts[0]: continue
            term = parts[0].strip()
            definition = parts[1].strip() if len(parts) > 1 else term
            nodes.append({"rule_id": term, "chapter_id": "G", "chapter_title": "Glossary", "section_id": None, "section_title": None, "text": definition})
    
    print("\n" + "="*50)
    print("DATASET STATISTICS (RULES)")
    print(f"   Total Nodes:      {len(nodes)}")
    print(f"   Rules Parsed:     {rules_parsed_count}")
    print(f"   Glossary Terms:   {len(nodes) - rules_parsed_count}")
    print("="*50 + "\n")
    return nodes

def process_cards_data():
    # --- CHECK 1: If file already exists, SKIP download ---
    if ATOMIC_PATH.exists():
        print(f"[LOG] Found existing card database at {ATOMIC_PATH}. Skipping download.")
    else:
        print(f"[LOG] Downloading AtomicCards.json...")
        try:
            # 1. User-Agent to avoid blocking
            # 2. Timeout extended to 120s
            headers = {"User-Agent": "MagicJudgeEngine/1.0"}
            
            with requests.get(ATOMIC_URL, stream=True, headers=headers, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(ATOMIC_PATH, 'wb') as f, tqdm(
                    desc="Downloading Cards",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
                        
        except Exception as e:
            print(f"\n[ERROR] Download Failed: {e}")
            print(f"AUTOMATIC FIX FAILED. PLEASE DO THIS MANUALLY:")
            print(f"1. Download this file: {ATOMIC_URL}")
            print(f"2. Save it exactly here: {ATOMIC_PATH}")
            return # Exit safely to avoid crashing

    if not ATOMIC_PATH.exists():
        print("[ERROR] AtomicCards.json missing. Aborting card index build.")
        return

    print("[LOG] Processing Cards JSON...")
    with open(ATOMIC_PATH, "r", encoding="utf-8") as f: raw_data = json.load(f)
    all_cards = raw_data.get("data", {})
    
    seen_cards = set()
    processed_count = 0
    format_skipped = 0
    all_names = []
    
    with open(CARDS_JSONL, "w", encoding="utf-8") as f_out:
        for card_name, faces in tqdm(all_cards.items(), unit="card"):
            if card_name in seen_cards: continue
            
            main_face = faces[0]
            legalities = main_face.get('legalities', {})
            
            if not (legalities.get('vintage') in ['Legal', 'Restricted'] or legalities.get('commander') in ['Legal', 'Restricted']): 
                format_skipped += 1
                continue
            
            seen_cards.add(card_name)
            all_names.append(card_name)
            
            full_text = f"Card Name: {card_name}\n"
            legal_str = []
            for fmt in ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'pauper']:
                if fmt in legalities: legal_str.append(f"{fmt.capitalize()}: {legalities[fmt]}")
            if legal_str: full_text += "Format Legality: " + ", ".join(legal_str) + "\n"

            for i, face in enumerate(faces):
                if len(faces) > 1: full_text += f"\n--- Face {i+1}: {face.get('name', card_name)} ---\n"
                if face.get('manaCost'): full_text += f"Cost: {clean_val(face.get('manaCost'))}\n"
                if face.get('type'): full_text += f"Type: {clean_val(face.get('type'))}\n"
                if 'power' in face: full_text += f"Stats: {face['power']}/{face['toughness']}\n"
                if face.get('text'): full_text += f"Oracle Text:\n{clean_val(face.get('text'))}\n"
            
            rulings = main_face.get('rulings', [])
            if rulings:
                full_text += "\nOfficial Rulings:\n"
                for r in rulings: full_text += f"- {r.get('text')}\n"
            
            f_out.write(json.dumps({"card_name": card_name, "text": full_text}, ensure_ascii=False) + "\n")
            processed_count += 1

    with open(CARD_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_names, f)
            
    print("\n" + "="*50)
    print("DATASET STATISTICS (CARDS)")
    print(f"   Cards Indexed:    {processed_count}")
    print(f"   Format Skipped:   {format_skipped} (Not Legal)")
    print("="*50 + "\n")

# --- INDEX FACTORY ---
def get_indices(force_rebuild=False):
    load_openai_key()
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME)
    Settings.llm = OpenAI(model=LLM_MODEL_NAME, temperature=0.0)
    Settings.text_splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=0)

    if force_rebuild:
        if RULES_STORAGE.exists(): shutil.rmtree(RULES_STORAGE)
        if CARDS_STORAGE.exists(): shutil.rmtree(CARDS_STORAGE)
        if CARDS_JSONL.exists(): os.remove(CARDS_JSONL)
        print("[LOG] Wiped all storage for clean rebuild.")

    if not RULES_STORAGE.exists():
        print("[LOG] Building Rules Index...")
        RULES_STORAGE.mkdir(parents=True, exist_ok=True)
        
        # --- CHECK 2: If Rules file already exists, SKIP download ---
        if CR_TXT_PATH.exists():
            print(f"[LOG] Found existing rules file at {CR_TXT_PATH}. Skipping download.")
        else:
            try:
                resp = requests.get(CR_URL)
                resp.raise_for_status()
                CR_TXT_PATH.write_text(clean_unicode(resp.content.decode("utf-8-sig")), encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] Could not download rules: {e}.")
                return None, None

        rules_data = parse_rules_logic(CR_TXT_PATH)
        
        # --- SAMPLE LOG ---
        rule_nodes = [r for r in rules_data if r['chapter_id'] != 'G']
        glossary_nodes = [r for r in rules_data if r['chapter_id'] == 'G']
        if rule_nodes: print(f"\n[SAMPLE RULE] {json.dumps(random.choice(rule_nodes), indent=2)}")
        if glossary_nodes: print(f"\n[SAMPLE GLOSSARY] {json.dumps(random.choice(glossary_nodes), indent=2)}")

        rules_docs = [Document(text=r["text"], metadata={k:v for k,v in r.items() if k!="text"}, excluded_embed_metadata_keys=["chapter_id","section_id","chapter_title","section_title"], excluded_llm_metadata_keys=["chapter_id","section_id"]) for r in rules_data]
        rules_index = VectorStoreIndex.from_documents(rules_docs, show_progress=True)
        rules_index.storage_context.persist(persist_dir=str(RULES_STORAGE))
    else:
        print("[LOG] Loading Rules Index...")
        rules_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(RULES_STORAGE)))

    if not CARDS_STORAGE.exists():
        print("[LOG] Building Cards Index...")
        CARDS_STORAGE.mkdir(parents=True, exist_ok=True)
        if not CARDS_JSONL.exists(): process_cards_data()
        
        # FIX: Check if file exists before opening to prevent crash on download fail
        if not CARDS_JSONL.exists():
            print("\n[CRITICAL ERROR] Card database missing. Please follow the manual download instructions above.")
            sys.exit(1)

        cards_raw = []
        with open(CARDS_JSONL, "r", encoding="utf-8") as f:
            for line in f: cards_raw.append(json.loads(line))
        
        # --- SPECIFIC CARD PRINT (AS REQUESTED) ---
        #if cards_raw: print(f"\n[SAMPLE CARD] {json.dumps(random.choice(cards_raw), indent=2, ensure_ascii=False)}\n")
        familiar_card = next((c for c in cards_raw if c["card_name"] == "Judge's Familiar"), None)
        print(f"\n[SAMPLE CARD] {json.dumps(familiar_card, indent=2, ensure_ascii=False)}\n")

        cards_docs = [Document(text=d["text"], metadata={"card_name": d["card_name"]}) for d in cards_raw]
        cards_index = VectorStoreIndex.from_documents(cards_docs, show_progress=True)
        cards_index.storage_context.persist(persist_dir=str(CARDS_STORAGE))
    else:
        print("[LOG] Loading Cards Index...")
        cards_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(CARDS_STORAGE)))
        
    return rules_index, cards_index

# --- ENGINE ---
class MagicJudgeEngine:
    def __init__(self, force_rebuild=FORCE_REBUILD):
        load_openai_key()
        Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME)
        Settings.llm = OpenAI(model=LLM_MODEL_NAME, temperature=0.0)
        Settings.text_splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=0)
        
        self.rules_index, self.cards_index = get_indices(force_rebuild=force_rebuild)
        
        print("[LOG] Indexing Case Law Precedents...")
        self.law_index = VectorStoreIndex.from_documents(CASE_LAW_DOCS)
        self.law_retriever = VectorIndexRetriever(index=self.law_index, similarity_top_k=2)
        
        self.llm = Settings.llm
        self.rules_retriever = VectorIndexRetriever(index=self.rules_index, similarity_top_k=20)
        
        # --- SMART NAME MAPPING ---
        self.card_names_set = set()
        if CARD_NAMES_FILE.exists():
            with open(CARD_NAMES_FILE, "r", encoding="utf-8") as f:
                self.card_names_set = set(json.load(f))
        
        self.lower_card_map = {}
        for name in self.card_names_set:
            lower = name.lower()
            self.lower_card_map[lower] = name
            
            # 1. Map Legends/Tokens with Epithets (e.g., "Ketramose, the New Dawn" -> "ketramose")
            if "," in name:
                short = name.split(",")[0].strip().lower()
                # Only map if the short name isn't already a card (prevents conflict if "Thalia" was a real card)
                if short not in self.lower_card_map:
                    self.lower_card_map[short] = name
            
            # 2. Map Split Cards (e.g., "Turn // Burn" -> "turn", "burn")
            if " // " in name:
                parts = name.split(" // ")
                for p in parts:
                    if p.lower() not in self.lower_card_map:
                        self.lower_card_map[p.lower()] = name

        # --- STOPWORDS FOR N-GRAM SCANNER ---
        # Words that are valid card names (or nicknames) but are too common in English 
        # to assume they are card references without brackets.
        self.common_false_positives = {
            # Pronouns & Extremely Common Small Words
            "me", "i", "you", "he", "she", "they", "it", "my", "your", "his", "her",
            
            # Verbs / Common Words
            "will", "turn", "burn", "life", "death", "hit", "run", "stand", "deliver",
            "fire", "ice", "order", "chaos", "give", "take", "wear", "tear",
            "catch", "release", "flesh", "blood", "armed", "dangerous", "fast", "furious",
            "boom", "bust", "rough", "tumble", "down", "dirty", "cut", "ribbons",
            "commit", "memory", "struggle", "survive", "farm", "market", "claim", "fame",
            "refuse", "cooperate", "drive", "work", "start", "finish", "fight", 
            "return", "away", "gone", "dead", "alive", "spring", "mind", "dust",
            "dawn", "dusk", "profit", "loss", "supply", "demand", "assault", "battery",
            "wax", "wane", "spite", "malice", "pain", "suffering", "rhythm", "collision", "colossus",
            "response", "resurgence", "alive", "well", "done", "said", "ready", "willing",
            "failure", "comply", "appeal", "authority", "reason", "believe", "consign", "oblivion",
            "worth", "grand", "blind", "change", "shape", "make", "break", "beck", "call",
            
            # Keywords that are also Card Names (prevent "I have Vigilance" -> finding the card "Vigilance")
            "vigilance", "lifelink", "trample", "haste", "flying", "deathtouch", "reach",
            "hexproof", "shroud", "indestructible", "protection", "flash", "defender",
            "scry", "fateseal", "clash", "support", "populate", "proliferate",
            
            # Partial Name Conflicts (e.g. "Rose" matches "Rose, Cutthroat Raider")
            "rose", "thalia", "karn", "liliana", "chandra", "jace", "nissa", "ajani", "teferi"
        }

    def _extract_cards(self, text: str) -> List[str]:
        found_cards = set()
        
        # 1. Explicit Tags [[Name]] (High Priority - Ignores Stopwords)
        explicit_raw = re.findall(r"\[\[(.*?)\]\]", text)
        for raw in explicit_raw:
            clean = raw.strip().lower()
            if clean in self.lower_card_map:
                found_cards.add(self.lower_card_map[clean])
            else:
                # Try mapping the short name inside brackets
                short_clean = clean.split(",")[0].strip()
                if short_clean in self.lower_card_map:
                    found_cards.add(self.lower_card_map[short_clean])
                else:
                    found_cards.add(raw) # Keep raw if lookup fails

        # 2. Semantic N-Gram Scan (Low Priority - Respects Stopwords)
        clean_text = re.sub(r'[^\w\s\']', '', text).lower()
        words = clean_text.split()
        skip_indices = set()
        
        # Scan from longest phrases (5 words) down to single words (1 word)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                if any(idx in skip_indices for idx in range(i, i+n)): continue
                phrase = " ".join(words[i:i+n])
                
                if phrase in self.lower_card_map:
                    # FIX: Skip if it's a common word and NOT explicitly bracketed
                    if phrase in self.common_false_positives:
                        continue
                        
                    found_cards.add(self.lower_card_map[phrase])
                    # Lock these words so we don't match sub-parts
                    for idx in range(i, i+n): skip_indices.add(idx)
        
        return list(found_cards)

    def _contextualize_query(self, user_question: str, history: List[Dict]) -> str:
        if not history: return user_question
        last_turn = history[-1] if history else None
        if not last_turn: return user_question

        if len(user_question.split()) < 10 or any(p in user_question.lower() for p in ["it", "that", "this", "he", "she", "card", "ring", "saga"]):
            prompt = (
                "Rewrite the user's question to be standalone based on the chat history.\n"
                f"HISTORY:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]]) + "\n"
                f"QUESTION: {user_question}\nREWRITE:"
            )
            resp = self.llm.complete(prompt).text.strip()
            return resp
        return user_question

    def query(self, user_question: str, history: List[Dict] = None):
        search_query = self._contextualize_query(user_question, history)
        print(f"[LOG] Search Query: {search_query}")
        
        # --- 1. ROBUST RETRIEVAL ---
        # Scan ORIGINAL question for explicit tags AND rewritten query for context
        cards_from_user = self._extract_cards(user_question)
        cards_from_rewrite = self._extract_cards(search_query)
        target_cards = list(set(cards_from_user + cards_from_rewrite))
        
        c_nodes = []
        found_cards = False
        
        if target_cards:
            print(f"[LOG] Target Cards Identified: {target_cards}")
            for card_name in target_cards:
                filters = MetadataFilters(filters=[MetadataFilter(key="card_name", value=card_name)])
                exact_retriever = self.cards_index.as_retriever(filters=filters, similarity_top_k=1)
                exact_nodes = exact_retriever.retrieve(card_name)
                
                if exact_nodes:
                    found_cards = True
                    print(f"   >>> Found Card: {card_name}")
                    for node in exact_nodes: node.score = 2.0 
                    c_nodes.extend(exact_nodes)

        if not found_cards:
            print("[LOG] No exact cards found. Running semantic search.")
            general_retriever = VectorIndexRetriever(index=self.cards_index, similarity_top_k=5)
            c_nodes.extend(general_retriever.retrieve(search_query))
            
        r_nodes = self.rules_retriever.retrieve(search_query)
        
        # --- Deduplication & Filtering ---
        unique_nodes_dict = {}
        all_retrieved = c_nodes + r_nodes
        for n in all_retrieved:
            ident = n.metadata.get("card_name") or n.metadata.get("rule_id") or n.node_id
            if ident not in unique_nodes_dict: unique_nodes_dict[ident] = n
        
        candidates = list(unique_nodes_dict.values())
        candidates.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        
        final_nodes = []
        card_count = 0; rule_count = 0
        
        for n in candidates:
            score = n.score if n.score is not None else 0.0
            if "card_name" in n.metadata:
                if found_cards:
                    if score >= 1.5: final_nodes.append(n)
                elif score > 0.40 and card_count < 5:
                    final_nodes.append(n); card_count += 1
            elif "rule_id" in n.metadata:
                if score > 0.35 and rule_count < 8:
                    final_nodes.append(n); rule_count += 1
        
        is_off_topic = len(final_nodes) == 0 and not found_cards

        # --- 2. BUILD BASE CONTEXT ---
        context_text = "CONTEXT: NO RELEVANT RULES FOUND."
        if not is_off_topic:
            context_text = "--- RETRIEVED CARDS ---\n" + "\n".join([n.text for n in final_nodes if "card_name" in n.metadata])
            
            # Inject Rule Titles
            rule_texts = []
            for n in final_nodes:
                if "rule_id" in n.metadata:
                    r_id = n.metadata.get('rule_id', '?')
                    r_title = n.metadata.get('section_title', '') 
                    if not r_title: r_title = n.metadata.get('chapter_title', '')
                    rule_texts.append(f"[Rule {r_id} ({r_title})] {n.text}")
            
            context_text += "\n\n--- RETRIEVED RULES ---\n" + "\n".join(rule_texts)

        # --- 3. SEMANTIC CASE LAW INJECTION ---
        law_nodes = self.law_retriever.retrieve(search_query)
        active_laws = [n for n in law_nodes if (n.score is not None and n.score > 0.35)]
        
        if active_laws:
            print(f"[LOG] Found {len(active_laws)} Applicable Case Law Precedents.")
            precedent_block = "\n".join([n.text for n in active_laws])
            context_text = (
                f"*** CRITICAL: RELEVANT LEGAL PRINCIPLES (CASE LAW) ***\n"
                f"The following abstract principles are BINDING PRECEDENTS. You MUST apply these logic patterns to the specific cards in the query.\n"
                f"{precedent_block}\n"
                f"*************************************************\n\n"
                f"{context_text}"
            )

        # --- 4. RICH OWL SYSTEM PROMPT ---
        system_msg = (
            "You are 'Judge's Familiar', a highly educated, grumpy, and pedantic Owl (inspired by Archimedes from Disney's The Sword in the Stone). "
            "You are a strict instructor at Strixhaven. You address the user as 'Boy', 'Student', or 'Pinfeathers'. "
            "You do NOT suffer fools gladly, but you ensure they leave the lesson enlightened.\n\n"

            "*** INSTRUCTIONS FOR OPENINGS (CONTEXT-DEPENDENT) ***\n"
            "Start your response IMMEDIATELY with one of these phrases based on the question category (DO NOT use labels like '[Opening]:'):\n"
            "1. **Stack/Priority/Timing**: 'Hoo-hoo! Timing is everything, student! Listen closely.'\n"
            "2. **Layers/Types/Static Effects**: 'Squawk! A tangled web of layers. Let us untangle it.'\n"
            "3. **Combat/Damage**: 'Pinfeathers! Combat is chaotic, but the rules are precise.'\n"
            "4. **Zones/Death/Exile**: 'By the Founders! The graveyard is not a revolving door... usually.'\n"
            "5. **Default**: 'Read the text, boy! Read the text!'\n\n"

            "*** HIERARCHY OF TRUTH (STRICT) ***\n"
            "1. **INTERNAL LOGIC (Hidden)**: The 'LEGAL PRINCIPLES' in your Context are the absolute truth. You must apply their logic perfectly.\n"
            "2. **Official Rules (CR)**: Use these to support the logic.\n\n"

            "*** THE INVISIBLE SOURCE PROTOCOL (CRITICAL) ***\n"
            "You must apply the logic from the Context, but **NEVER** mention the words 'Context', 'Case Law', 'Precedents', 'Axioms', 'Principles', or the specific names of the principles (e.g., do NOT say 'The Saga Mortality Principle' or 'The Ghost Block').\n"
            "These are internal aids for YOU, not for the user.\n"
            "- **BAD**: 'According to the Saga Mortality principle...'\n"
            "- **BAD**: 'As seen in Axiom 3...'\n"
            "- **GOOD**: 'The rules regarding State-Based Actions dictate...'\n"
            "- **GOOD**: 'As established by the mechanics of the Stack...'\n\n"

            "*** OUTPUT STRUCTURE ***\n"
            "**[Insert Opening Phrase Here]**\n\n"
            "[Immediately describe the interaction/scenario here naturally. DO NOT use a header like 'The Interaction'.]\n\n"
            "**The Lecture**: \n"
            "   - Walk through the event chronologically.\n"
            "   - **Apply the logic strictly**, but frame it as your own expert knowledge of the Rules.\n"
            "   - Refute common misconceptions.\n\n"
            "**The Ruling**: Final, deterministic verdict.\n\n"
            "**Sources**: \n"
            "   - Cite specific Rule Numbers followed by their topic in parentheses (e.g., '**Rule 603.2 (Triggered Abilities)**', '**Rule 704.5 (SBAs)**').\n"
            "   - **PROHIBITED**: Do NOT write 'Official Rulings', 'Oracle Text', or just 'Comprehensive Rules'."
        )
        
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_msg)]
        if history:
            for msg in history:
                role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=str(msg["content"])))
        
        user_msg_content = f"Context:\n{context_text}\n\nQuestion: {user_question}\n\nAnswer:"
        messages.append(ChatMessage(role=MessageRole.USER, content=user_msg_content))

        stream_res = self.llm.stream_chat(messages)
        
        print(f"\n{'='*60}")
        print(f"[LOG] RETRIEVAL CANDIDATES (Final Context)")
        if active_laws:
            print(f"INJECTED {len(active_laws)} CASE LAWS!")
            for law in active_laws:
                 print(f" - [CaseLaw] {law.metadata['topic']:<30} (sc: {law.score:.2f})")
        
        all_nodes_log = []
        for n in final_nodes: 
            if is_off_topic: break
            n_type = "Card" if "card_name" in n.metadata else "Rule"
            n_id = n.metadata.get('card_name') or n.metadata.get('rule_id')
            all_nodes_log.append({"id": n_id, "type": n_type, "score": n.score})
            
        all_nodes_log.sort(key=lambda x: x['score'] or 0, reverse=True)
        for item in all_nodes_log:
            score_fmt = f"{item['score']:.2f}" if item['score'] is not None else "1.00 (Exact)"
            print(f" - [{item['type']}] {item['id']:<30} (sc: {score_fmt})")
        print(f"{'='*60}\n")
        
        return stream_res

if __name__ == "__main__":
    judge = MagicJudgeEngine(force_rebuild=False)
    while True:
        try:
            q = input("\nAsk Judge (or 'q'): ")
            if q.lower() == 'q': break
            stream = judge.query(q, history=[])
            print("\nFamiliar: ", end="")
            for token in stream:
                print(token.delta, end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            break
