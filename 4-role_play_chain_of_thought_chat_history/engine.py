import os
import re
import json
import shutil
import random
import unicodedata
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
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

# Set to False (Index assumed built)
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

# --- SETUP ---
def load_openai_key():
    if os.environ.get("OPENAI_API_KEY"): return
    load_dotenv()
    if os.environ.get("OPENAI_API_KEY"): return
    load_dotenv(Path("..") / ".env")

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
    
    # --- STATISTICS LOG (RULES) ---
    print("\n" + "="*50)
    print("DATASET STATISTICS (RULES)")
    print(f"   Total Nodes:      {len(nodes)}")
    print(f"   Rules Parsed:     {rules_parsed_count}")
    print(f"   Glossary Terms:   {len(nodes) - rules_parsed_count}")
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
    
    with open(CARDS_JSONL, "w", encoding="utf-8") as f_out:
        for card_name, faces in tqdm(all_cards.items(), unit="card"):
            if card_name in seen_cards: continue
            
            main_face = faces[0]
            legalities = main_face.get('legalities', {})
            
            if not (legalities.get('vintage') in ['Legal', 'Restricted'] or legalities.get('commander') in ['Legal', 'Restricted']): 
                format_skipped += 1
                continue
            
            seen_cards.add(card_name)
            
            full_text = f"Card Name: {card_name}\n"
            
            legal_str = []
            for fmt in ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'pauper']:
                if fmt in legalities:
                    legal_str.append(f"{fmt.capitalize()}: {legalities[fmt]}")
            if legal_str:
                full_text += "Format Legality: " + ", ".join(legal_str) + "\n"

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
            
    # --- STATISTICS LOG (CARDS) ---
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
        if CARDS_JSONL.exists(): 
            os.remove(CARDS_JSONL)
            print("[LOG] Deleted old JSONL file to ensure clean ingestion.")
        print("[LOG] Wiped all storage for clean rebuild.")

    if not RULES_STORAGE.exists():
        print("[LOG] Building Rules Index...")
        RULES_STORAGE.mkdir(parents=True, exist_ok=True)
        if not CR_TXT_PATH.exists():
            resp = requests.get(CR_URL); CR_TXT_PATH.write_text(clean_unicode(resp.content.decode("utf-8-sig")), encoding="utf-8")
        rules_data = parse_rules_logic(CR_TXT_PATH)
        
        # --- SAMPLE LOG (RULES) ---
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
        
        # --- SAMPLE LOG (CARDS) ---
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
        self.rules_retriever = VectorIndexRetriever(index=self.rules_index, similarity_top_k=8)
        self.cards_index = self.cards_index

    def _extract_explicit_cards(self, query: str) -> List[str]:
        return re.findall(r"\[\[(.*?)\]\]", query)

    def _contextualize_query(self, user_question: str, history: List[Dict]) -> str:
        if not history: return user_question
        last_turn = history[-1] if history else None
        if not last_turn: return user_question

        if len(user_question.split()) < 10 or any(p in user_question.lower() for p in ["it", "that", "this", "he", "she", "card"]):
            prompt = (
                "Rewrite the user's question to be standalone based on the chat history.\n"
                f"HISTORY:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]]) + "\n"
                f"QUESTION: {user_question}\nREWRITE:"
            )
            resp = self.llm.complete(prompt).text.strip()
            print(f"[LOG] Rewritten Query: {resp}")
            return resp
        return user_question

    def query(self, user_question: str, history: List[Dict] = None):
        search_query = self._contextualize_query(user_question, history)
        print(f"[LOG] Search Query: {search_query}")
        
        explicit_cards = list(set(self._extract_explicit_cards(search_query)))
        c_nodes = []
        found_explicit_cards = False
        
        if explicit_cards:
            print(f"[LOG] Explicit Mode Active: {explicit_cards}")
            for card_name in explicit_cards:
                filters = MetadataFilters(filters=[MetadataFilter(key="card_name", value=card_name)])
                exact_retriever = self.cards_index.as_retriever(filters=filters, similarity_top_k=1)
                exact_nodes = exact_retriever.retrieve(card_name)
                
                if exact_nodes:
                    found_explicit_cards = True
                    print(f"   >>> Found Exact Match: {card_name}")
                    for node in exact_nodes: node.score = 1.0 
                    c_nodes.extend(exact_nodes)
                else:
                    print(f"   >>> Failed Exact Match for '{card_name}'. Falling back.")
                    fallback_retriever = VectorIndexRetriever(index=self.cards_index, similarity_top_k=3)
                    c_nodes.extend(fallback_retriever.retrieve(card_name))
        
        # EXCLUSIVE EXPLICIT MODE
        if not found_explicit_cards:
            general_retriever = VectorIndexRetriever(index=self.cards_index, similarity_top_k=3)
            c_nodes.extend(general_retriever.retrieve(search_query))
            
        r_nodes = self.rules_retriever.retrieve(search_query)
        
        # --- DEDUPLICATION ---
        unique_nodes_dict = {}
        all_retrieved = c_nodes + r_nodes
        
        for n in all_retrieved:
            ident = n.metadata.get("card_name") or n.metadata.get("rule_id") or n.node_id
            if ident not in unique_nodes_dict:
                unique_nodes_dict[ident] = n
        
        candidates = list(unique_nodes_dict.values())
        candidates.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        
        # --- STRICT FILTERING (Logic Applied) ---
        final_nodes = []
        card_count = 0
        rule_count = 0
        
        for n in candidates:
            score = n.score if n.score is not None else 0.0
            
            if "card_name" in n.metadata:
                if found_explicit_cards:
                    if score >= 0.99: final_nodes.append(n)
                elif score > 0.40 and card_count < 3:
                    final_nodes.append(n)
                    card_count += 1
            elif "rule_id" in n.metadata:
                if score > 0.40 and rule_count < 3:
                    final_nodes.append(n)
                    rule_count += 1
        
        best_score = max((n.score for n in final_nodes if n.score is not None), default=0.0)
        is_off_topic = len(final_nodes) == 0 and not found_explicit_cards

        context = "CONTEXT: NO RELEVANT RULES FOUND."
        if not is_off_topic:
            context = "--- CARDS ---\n" + "\n".join([n.text for n in final_nodes if "card_name" in n.metadata])
            context += "\n\n--- RULES ---\n" + "\n".join([f"[Rule {n.metadata.get('rule_id', '?')}] {n.text}" for n in final_nodes if "rule_id" in n.metadata])

        # --- FINAL PROMPT ---
        system_msg = (
            "You are 'Judge's Familiar', a scholarly, pedantic, but incredibly helpful Owl who serves as a Magic: The Gathering rules assistant. "
            "You speak with the academic air of Archimedes (from The Sword in the Stone). "
            "You love teaching rule interactions and explaining the 'Why' behind every result. "
            "You do NOT give brief answers; you give lessons.\n\n"
            
            "*** MANDATE ***\n"
            "Provide strictly accurate rulings based *only* on the provided context.\n"
            "If the context is 'NO RELEVANT RULES FOUND', politely decline to answer off-topic questions, perhaps making a bird pun.\n\n"

            "*** HIDDEN REASONING PROCESS (CHAIN OF THOUGHT) ***\n"
            "Before answering, you MUST output a block starting with '### THOUGHTS ###'.\n"
            "In this block, explicitly step through these checks:\n"
            "1. **Identify Objects**: What cards are involved? Resolve 'it' or 'them' using history.\n"
            "2. **Check Legality**: Is the card banned? (Only relevant if the user asks about format legality).\n"
            "3. **Apply Protocols**: Go through the Protocols below step-by-step.\n"
            "   - **CRITICAL CHECK**: If a source moves/leaves the zone, use Protocol 4 (the Snapshot rule).\n"
            "   - **Check for Source Removal**: If source is gone, does ability fizzle? (Answer: NO. Apply Protocol 4).\n"
            "4. **Formulate Conclusion**: Determine the final game state.\n"
            "Close the block with '### FINAL ANSWER ###'.\n\n"

            "*** RULES ENGINE PROTOCOLS: Apply in ORDER. Use the logic, but NEVER mention the internal names of these protocols (e.g. do not say 'Saga Survival Doctrine' or 'Layer System Protocol'). Just explain the interaction naturally. ***\n"
            "1. **THE GOLDEN RULE**: Specific card text overrides general game rules (Rule 101.1). "
            "Crucially, **'Can't' overrides 'Can' or 'Does'**.\n\n"

            "2. **THE LAYER SYSTEM PROTOCOL (Rule 613)**:\n"
            "   - If continuous effects interact, apply them in Layer order.\n"
            "   - **CRITICAL INTERACTION (Rule 305.7)**: If an effect turns a land into a Basic Land type (Layer 4), it loses abilities in its *printed* text box.\n"
            "   - **HOWEVER**: It DOES NOT lose abilities granted to it by other resolved effects or triggers (which apply in Layer 6).\n"
            "   - *Example*: A Land Saga transformed by [[Blood Moon]] loses printed chapters (Layer 4) but keeps abilities gained from resolved Chapter triggers (Layer 6).\n\n"

            "3. **THE SAGA SURVIVAL DOCTRINE (Rule 704.5s)**: \n"
            "   - A Saga is sacrificed ONLY if it meets two conditions: (A) Counters >= Max Chapters AND (B) It has 'one or more chapter abilities'.\n"
            "   - **Logic Check**: If the Layer System (Protocol 2) determines the Saga lost its *printed* chapter abilities (e.g. via [[Blood Moon]]), Condition (B) FAILS.\n"
            "   - **Crucial Distinction**: Non-printed abilities (like those gained through a previous chapter trigger) are NOT lost (Layer 6) and remain available.\n"
            "   - **Result**: The Saga is **NOT** sacrificed. It remains on the battlefield.\n\n"

            "4. **THE 'GRENADE' DOCTRINE & THE SNAPSHOT RULE (Rule 113.7a, 608.2h)**: \n"
            "   - Abilities on the stack exist **independently** of their source (like a grenade thrown by a soldier).\n"
            "   - If the source is removed (exiled/destroyed) before the ability resolves, the ability **STILL RESOLVES**. It does **NOT** fizzle.\n"
            "   - **THE SNAPSHOT RULE (LAST KNOWN INFORMATION)**: When an ability needs to know information about a source that is gone (e.g. 'how many counters are on it'), the game takes a **Snapshot** of the card as it existed **the moment before** it left the battlefield.\n"
            "   - **MANDATORY LOGIC**: The ability sees the valid information from the Snapshot (counters, power, etc.) and functions normally. It does NOT see a blank or empty object. It treats the source as having the characteristics it had just before leaving.\n"
            "   - **EXAMPLE**: If a source with counters is exiled with its ability on the stack, the ability resolves using the number of counters it had before leaving.\n\n"

            "5. **THE 'WITNESS' DOCTRINE (Rule 603.6c)**:\n"
            "   - Leaves-the-battlefield abilities look back in time.\n"
            "   - If a permanent triggers when something leaves the battlefield (including itself), it **DOES TRIGGER** for its own exit.\n"
            "   - Example: If [[Ketramose]] or [[The Gitrog Monster]] is exiled, it sees itself leave and triggers.\n\n"
            
            "6. **THE LITERAL TEXT DOCTRINE**:\n"
            "   - Read the provided Oracle Text literally. Do not infer keywords that are not written.\n"
            "   - (e.g., [[Solitude]] exiles a creature; it does NOT grant Shroud/Hexproof unless the text says so).\n\n"

            "*** OUTPUT INSTRUCTIONS (FOR FINAL ANSWER) ***\n"
            "1. **Format**: \n"
            "   ### THOUGHTS ###\n"
            "   (Your hidden logic)\n"
            "   ### FINAL ANSWER ###\n"
            "   (Your actual response to the user)\n"
            "2. **Tone**: Scholarly, Educational, and Comprehensive. DO NOT BE CONCISE. Explain the reasoning steps clearly so the user learns the rules.\n"
            "3. **Persona**: Use the owl persona. You may use phrases like 'Hoo!', 'A fascinating interaction!', or 'Let us consult the scrolls.'\n"
            "4. **Content**: Include the legality check ONLY if the user asked about formats. Otherwise, focus on the gameplay interaction.\n"
            "5. **Citations**: End with 'Sources: [Rule IDs / [[Card Names]]]'. **IMPORTANT: Only cite Rules and Cards that appear in the provided CONTEXT.** Do not cite rules from your internal protocols if they are not in the context. If the context is 'NO RELEVANT RULES FOUND', DO NOT print a Sources line."
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
        # Log unique nodes only
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
