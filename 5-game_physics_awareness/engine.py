import os
import re
import json
import shutil
import random
import unicodedata
import requests
import time
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
LLM_MODEL_NAME = "gpt-4o-mini"
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

# --- CORE GAME PHYSICS AXIOMS ---
CORE_GAME_ENGINE = """
*** THE MAGIC: THE GATHERING GAME PHYSICS ENGINE ***
Apply these internal logic gates STRICTLY. Do NOT cite "Axioms" in the final output; cite the Rules provided here.

AXIOM 1: LAYERS & ABILITY RETENTION (Rule 613)
   - **Type-Changing (Layer 4)**: Effects that change a permanent's subtype (e.g. Blood Moon, Song of the Dryads) wipe all abilities in the *printed* text box.
   - **Gained Abilities (Layer 6)**: Abilities granted by *other* resolved effects (like a previous Saga Chapter trigger) are NOT removed by Layer 4 type changes.
   - **The Consequence**: If a Saga is on Chapter II, it has ALREADY GAINED the ability from Chapter I and Chapter II. A Layer 4 effect makes it a Mountain, but it **RETAINS** the abilities it gained (Layer 6). It can still tap for mana (Chapter I) or make tokens (Chapter II).

AXIOM 2: SAGA MORTALITY CHECK (Rule 704.5s)
   - A Saga is sacrificed ONLY if: 
     1. The number of lore counters >= The final chapter number.
     2. AND the Saga actually has one or more chapter abilities.
   - **Logic Gate**: If Layer 4 (Blood Moon) removed the *printed* chapter abilities, the Saga currently has 0 chapter abilities. Condition 2 FAILS. The Saga is NOT sacrificed. It survives in stasis.

AXIOM 3: THE INDEPENDENCE OF ABILITIES (Rule 113.7a - The Stack)
   - Once an ability is activated or triggered and placed on the stack, it exists independently of its source.
   - **The Grenade Rule**: Removing the source (e.g. exiling The One Ring) does NOT stop the ability from resolving. It simply uses Last Known Information (LKI).

AXIOM 4: MATH & LKI (Rule 608.2h)
   - **LKI Override**: If an ability needs info from a source that is gone (e.g., "draw cards for each counter on this"), use the state of the source **immediately before** it left the battlefield. Do NOT use Rule 607 to say it fails.
   - **Multiplier Check**: If an effect says "deal X damage for *each* Y", you MUST multiply.
     - *Example*: "Deal 2 damage for each card" with 2 cards = **4 damage total**.
"""

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
    print("="*50 + "\n")
    return nodes

def process_cards_data():
    if not ATOMIC_PATH.exists():
        print(f"[LOG] Downloading AtomicCards.json...")
        with requests.get(ATOMIC_URL, stream=True) as r:
            with open(ATOMIC_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
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
        if not CR_TXT_PATH.exists():
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
        
        cards_raw = []
        with open(CARDS_JSONL, "r", encoding="utf-8") as f:
            for line in f: cards_raw.append(json.loads(line))
        
        if cards_raw: print(f"\n[SAMPLE CARD] {json.dumps(random.choice(cards_raw), indent=2, ensure_ascii=False)}\n")

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
        self.llm = Settings.llm
        self.rules_retriever = VectorIndexRetriever(index=self.rules_index, similarity_top_k=20)
        
        self.card_names_set = set()
        if CARD_NAMES_FILE.exists():
            with open(CARD_NAMES_FILE, "r", encoding="utf-8") as f:
                self.card_names_set = set(json.load(f))
        self.lower_card_map = {name.lower(): name for name in self.card_names_set}

    def _extract_cards(self, query: str) -> List[str]:
        explicit = re.findall(r"\[\[(.*?)\]\]", query)
        if explicit: return list(set(explicit))
        
        potential_cards = []
        clean_query = re.sub(r'[^\w\s\']', '', query).lower()
        words = clean_query.split()
        skip_indices = set()
        
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                if any(idx in skip_indices for idx in range(i, i+n)): continue
                phrase = " ".join(words[i:i+n])
                if phrase in self.lower_card_map:
                    potential_cards.append(self.lower_card_map[phrase])
                    for idx in range(i, i+n): skip_indices.add(idx)
        return list(set(potential_cards))

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
        
        target_cards = self._extract_cards(search_query)
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
        context = "CONTEXT: NO RELEVANT RULES FOUND."
        if not is_off_topic:
            context = "--- CARDS ---\n" + "\n".join([n.text for n in final_nodes if "card_name" in n.metadata])
            context += "\n\n--- RULES ---\n" + "\n".join([f"[Rule {n.metadata.get('rule_id', '?')}] {n.text}" for n in final_nodes if "rule_id" in n.metadata])

        # --- FINAL PROMPT ---
        system_msg = (
            "You are 'Judge's Familiar', a scholarly, pedantic Owl who serves as a strict instructor at Strixhaven. "
            "You speak with an academic, slightly haughty, but ultimately helpful tone. "
            "You DO NOT simply list facts; you *teach* the rules, often reminding the user of basic principles they should already know.\n"
            "Avoid repetitive greetings. Instead of always starting with 'Hoot!', vary your openings: express intrigue, correct a misconception, or simply dive into the complexity of the stack.\n\n"
            
            "*** MANDATE ***\n"
            "Provide strictly accurate rulings based *only* on the provided context and the GAME PHYSICS AXIOMS below.\n"
            f"{CORE_GAME_ENGINE}\n\n"

            "*** OUTPUT STRUCTURE ***\n"
            "Structure your answer in this exact format to minimize latency and maximize clarity:\n\n"
            
            "**[Insert a short, varied, in-character opening sentence]**\n\n"
            
            "### 1. The Interaction\n"
            "Identify the key cards and the specific mechanical conflict.\n\n"
            
            "### 2. The Logic (Step-by-Step)\n"
            "Walk through the interaction chronologically. Apply the AXIOMS here explicitly. Inject your persona here (e.g., 'Observe that...', 'Crucially...', 'Do not be fooled by...').\n"
            "- For Layer Conflicts: Explain Layer 4 vs Layer 6. Note that gained abilities are RETAINED.\n"
            "- For Sacrifice: Explain the specific conditions for Saga death (Counters vs Chapters).\n"
            "- For Stack: Explain Independence of Abilities.\n\n"
            
            "### 3. The Ruling\n"
            "State the final result clearly in one sentence.\n\n"
            
            "### Sources\n"
            "List citations as: '[[Card Name]], [Rule ID (Rule Name)]'.\n"
            "You may cite rules found in the retrieved Context OR the GAME PHYSICS AXIOMS."
        )
        
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_msg)]
        if history:
            for msg in history:
                role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=str(msg["content"])))
        
        user_msg_content = f"Context:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
        messages.append(ChatMessage(role=MessageRole.USER, content=user_msg_content))

        stream_res = self.llm.stream_chat(messages)
        
        print(f"\n{'='*60}")
        print(f"[LOG] RETRIEVAL CANDIDATES (After Filtering)")
        all_nodes_log = []
        for n in final_nodes: 
            if is_off_topic: break
            n_type = "Card" if "card_name" in n.metadata else "Rule"
            n_id = n.metadata.get('card_name') or n.metadata.get('rule_id')
            all_nodes_log.append({"id": n_id, "type": n_type, "score": n.score})
            
        all_nodes_log.sort(key=lambda x: x['score'] or 0, reverse=True)
        for item in all_nodes_log:
            score_fmt = f"{item['score']:.2f}" if item['score'] else "1.00 (Exact)"
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
