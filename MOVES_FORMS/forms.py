"""
update_forms.py
~~~~~~~~~~~~~~~
Complementa moves em pokemonforms.txt (PBS de formas) consultando a PokéAPI,
ajusta o campo FormName e grava arquivos de saída.

python update_forms.py caminho/para/pokemonforms.txt
"""

from __future__ import annotations
import re, csv, sys, time, queue, logging, requests
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# ------------------------------------------------------------------ CONFIG
POKEAPI            = "https://pokeapi.co/api/v2"
VERSION_GROUP      = None           # None = pega learnset de TODAS as gerações
REQUESTS_PER_SEC   = 5
SLEEP_IF_RATELIMIT = 30
LOGLEVEL           = logging.INFO

IGNORE_MOVES_FILE  = "ignore_moves.txt"   # mesmo formato do script principal
FORMS_LIST_FILE    = "forms_list.txt"     # lista/mapeamento de nomes de forma
# -------------------------------------------------------------------------


############################################################################
# utilidades genéricas (mesmas do script principal)
############################################################################
log = logging.getLogger("update_forms")
logging.basicConfig(level=LOGLEVEL, format="%(levelname)s: %(message)s")

def normalize_move(move: str) -> str:
    return move.replace("-", "").replace(" ", "").upper()

def parse_moves_field(value: str) -> List[Tuple[int, str]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [(int(parts[i]), parts[i + 1].upper()) for i in range(0, len(parts), 2)]

def stringify_moves_field(moves: List[Tuple[int, str]]) -> str:
    moves_sorted = sorted(moves, key=lambda x: (x[0], x[1]))
    return ",".join(f"{lvl},{name}" for lvl, name in moves_sorted)

def parse_simple_list(value: str) -> List[str]:
    return [normalize_move(m) for m in value.split(",") if m.strip()]

def stringify_simple_list(lst) -> str:
    return ",".join(sorted(lst))


############################################################################
# 1) lista de golpes a ignorar (igual ao script 2)
############################################################################
def _load_ignore_moves() -> set[str]:
    p = Path(__file__).with_name(IGNORE_MOVES_FILE)
    if not p.exists():
        return set()
    return {normalize_move(l.strip())
            for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()}

IGNORE_MOVES = _load_ignore_moves()


############################################################################
# 2) lista/mapa de formas   (NOVO)
############################################################################
def _slug_from_line(line: str) -> Tuple[str, str]:
    """
    Converte uma linha de forms_list.txt em (FormNameOriginal, SLUG_CANÔNICO).
    Aceita dois formatos:
        Alolan                -> slug gerado automaticamente (Alola)
        Alolan = Alola        -> slug explícito fornecido
    """
    if "=" in line:
        left, right = map(str.strip, line.split("=", 1))
        return left, right

    original = line.strip()
    s = re.sub(r"\(.*?\)", "", original).strip()      
    s = re.sub(r"\b(Form|Forma|Forme|Mode|Style)\b", "", s, flags=re.I).strip()
    # regionais: Alolan→Alola, Galarian→Galar, Hisuian→Hisui
    regional_sub = {"ALOLAN": "ALOLA", "GALARIAN": "GALAR", "HISUIAN": "HISUI"}
    if s.upper() in regional_sub:
        s = regional_sub[s.upper()]
    elif s.upper().endswith("IAN"):
        s = s[:-3]      # Galarian→Galar
    elif s.upper().endswith("AN"):
        s = s[:-1]      # Alolan→Alola
    s = re.sub(r"\s+", " ", s).title()          
    slug = s.replace(" ", "-").upper()          # ‘Shadow Rider’→‘SHADOW-RIDER’
    return original, slug

def _load_forms_map() -> dict[str, str]:
    """
    Retorna {FormNameOriginal_em_PBS → SLUG}.
    Se um nome de forma não estiver no mapa, slugifica automaticamente.
    """
    p = Path(__file__).with_name(FORMS_LIST_FILE)
    forms_map: dict[str, str] = {}
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):     
                continue
            k, v = _slug_from_line(line)
            forms_map[k.upper()] = v
    return forms_map

FORMS_MAP = _load_forms_map()


############################################################################
# 3)  Rate‑limit simples para a PokéAPI
############################################################################
session = requests.Session()
_last_req = 0.0
_tokens   = queue.Queue(maxsize=REQUESTS_PER_SEC)
for _ in range(REQUESTS_PER_SEC):
    _tokens.put_nowait(None)

def _rate_limit():
    global _last_req
    try:
        _tokens.get_nowait()
    except queue.Empty:
        dt = time.time() - _last_req
        time.sleep(max(0, 1 - dt))
    finally:
        _last_req = time.time()
        _tokens.put_nowait(None)

def fetch_pokemon(api_name: str) -> Optional[dict]:
    url = f"{POKEAPI}/pokemon/{api_name}"
    _rate_limit()
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        log.warning("HTTP 429 – aguardando %ss", SLEEP_IF_RATELIMIT)
        time.sleep(SLEEP_IF_RATELIMIT)
        return fetch_pokemon(api_name)
    r.raise_for_status()
    return r.json()


############################################################################
# 4)  learnset (igual ao script 1, mas sem deduplicar “menor nível”)
############################################################################
def extract_moves(p_data: dict) -> dict[str, set]:
    """
    Retorna:
        {
          "level": {(nível, MOVE)},   # cada MOVE aparece só 1x (menor nível)
          "tutor": {MOVE, ...},
          "egg"  : {MOVE, ...}
        }
    """
    level_min: dict[str, int] = {}     
    tutor, egg = set(), set()

    for entry in p_data["moves"]:
        move_norm = normalize_move(entry["move"]["name"])

        for det in entry["version_group_details"]:
            if VERSION_GROUP and det["version_group"]["name"] != VERSION_GROUP:
                continue

            method = det["move_learn_method"]["name"]
            if method == "level-up":
                lvl = det["level_learned_at"] or 1
                prev = level_min.get(move_norm)
                if prev is None or lvl < prev:
                    level_min[move_norm] = lvl

            elif method == "tutor":
                tutor.add(move_norm)

            elif method == "egg":
                egg.add(move_norm)

    level_set = {(lvl, mv) for mv, lvl in level_min.items()}
    return {"level": level_set, "tutor": tutor, "egg": egg}


############################################################################
# 5)  Classe PBSBlock adaptada ao pokemonforms.txt
############################################################################
SEP_RE  = re.compile(r"^#-+\s*$", re.M)
FIELD_RE = re.compile(r"^(?P<k>\w+)\s*=\s*(?P<v>.*)$")

class FormBlock:
    def __init__(self, text: str):
        self.text = text
        self.fields = {}
        for ln in text.splitlines():
            m = FIELD_RE.match(ln)
            if m:
                self.fields[m["k"]] = m["v"]

    @property
    def species(self) -> str:
        """‘RATICATE’ a partir de ‘[RATICATE,1]’."""
        m = re.match(r"\[(?P<sp>[^,\]]+)", self.text.strip())
        return m["sp"] if m else ""

    @property
    def form_name(self) -> str:
        return self.fields.get("FormName", "")

    def replace_field(self, key: str, value: str):
        pat = re.compile(rf"^{key}\s*=.*$", re.M)
        repl = f"{key} = {value}"
        if pat.search(self.text):
            self.text = pat.sub(repl, self.text, 1)
        else:
            self.text = self.text.rstrip() + f"\n{repl}\n"

    # ---------- atualização de moves (igual ao PBSBlock.update_moves) -----
    def update_moves(self, nlvl, ntutor, negg):
        changes = {"level": [], "tutor": [], "egg": []}

        # level
        cur_lvl = parse_moves_field(self.fields.get("Moves", "")) if self.fields.get("Moves") else []
        cur_names = {normalize_move(n) for _, n in cur_lvl}
        add_lvl = {(l, n) for (l, n) in nlvl if n not in cur_names and n not in IGNORE_MOVES}
        if add_lvl:
            cur_lvl += add_lvl
            self.replace_field("Moves", stringify_moves_field(cur_lvl))
            changes["level"] = [n for _, n in sorted(add_lvl)]

        # tutor
        cur_tutor = set(parse_simple_list(self.fields.get("TutorMoves", "")))
        add_tutor = {m for m in ntutor if m not in cur_tutor and m not in IGNORE_MOVES}
        if add_tutor:
            self.replace_field("TutorMoves", stringify_simple_list(cur_tutor | add_tutor))
            changes["tutor"] = sorted(add_tutor)

        # egg
        cur_egg = set(parse_simple_list(self.fields.get("EggMoves", "")))
        add_egg = {m for m in negg if m not in cur_egg and m not in IGNORE_MOVES}
        if add_egg:
            self.replace_field("EggMoves", stringify_simple_list(cur_egg | add_egg))
            changes["egg"] = sorted(add_egg)

        return changes


############################################################################
# 6)  Pipeline
############################################################################
def slug_for_form(form_name_raw: str) -> str:
    """Converte ‘Alolan’, ‘Shadow Rider’, … → slug (‘ALOLA’, ‘SHADOW-RIDER’)."""
    key = form_name_raw.strip().upper()
    return FORMS_MAP.get(key) 

def process_block(txt: str) -> tuple[str, Optional[dict[str, list[str]]]]:
    """
    Processa um bloco do pokemonforms.txt.

    • Só mexe em formas listadas em forms_list.txt (FORMS_MAP).
    • NÃO altera o valor original de FormName no arquivo.
    • Complementa Moves / TutorMoves / EggMoves caso a forma exista
      na PokéAPI.
    """
    blk = FormBlock(txt)

    if not blk.species:
        return txt, None

    # 1) apenas formas presentes em forms_list.txt
    slug = slug_for_form(blk.form_name)
    if slug is None:
        return txt, {}               

    # 2) identifica a forma na PokéAPI
    api_name = f"{blk.species.lower()}-{slug.lower()}"
    data = fetch_pokemon(api_name)
    if data is None:
        return txt, None              # não encontrada → forms_ignore.txt

    # 3) complementa movimentos (FormName permanece como está)
    movesets = extract_moves(data)
    changes = blk.update_moves(
        movesets["level"], movesets["tutor"], movesets["egg"]
    )

    return blk.text, changes




def main(path: str | Path):
    path = Path(path)
    src = path.read_text(encoding="utf-8", errors="ignore")

    blocks = SEP_RE.split(src)
    seps   = SEP_RE.findall(src) + [""]

    out_blocks, ignores, changelog = [], [], {}

    for blk_txt in blocks:
        new_txt, chg = process_block(blk_txt)
        out_blocks.append(new_txt)
        if chg is None:
            ignores.append(FormBlock(blk_txt).species + " | " + FormBlock(blk_txt).form_name)
        elif any(chg.values()):
            key = FormBlock(new_txt).species + " - " + FormBlock(new_txt).form_name
            changelog[key] = chg

    out_path = path.parent / "forms_output.txt"
    out_path.write_text("".join(s + b for s, b in zip(seps, out_blocks)), encoding="utf-8")

    (path.parent / "forms_ignore.txt").write_text("\n".join(ignores), encoding="utf-8")

    with (path.parent / "forms_changelog.txt").open("w", encoding="utf-8") as fh:
        for k, chg in sorted(changelog.items()):
            fh.write(f"{k}:\n")
            for cat in ("level", "tutor", "egg"):
                if chg[cat]:
                    fh.write(f"  + {cat}: {', '.join(chg[cat])}\n")
            fh.write("\n")

    log.info("Processo concluído – arquivos gerados na pasta %s", path.parent)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python update_forms.py caminho/para/pokemonforms.txt")
    main(sys.argv[1])
