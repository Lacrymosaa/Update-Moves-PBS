"""
complete_moves.py
~~~~~~~~~~~~~~~~~
Preenche os movimentos faltantes no arquivo **pokemon.txt** (PBS) usando a
PokéAPI v2. Gera três saídas na mesma pasta do script:

* **output.txt**   – pokemon.txt completo (mantém formatação e comentários)
* **ignore.txt**   – lista de InternalNames que não puderam ser consultados
* **changelog.txt** – relatório das adições feitas em cada Pokémon

Requisitos
----------
python -m pip install requests tqdm

Uso
----
python complete_moves.py pokemon.txt
"""
from __future__ import annotations

import re
import sys
import json
import time
import queue
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable, Optional
import concurrent.futures as cf

import requests
from tqdm import tqdm


# ---------- Configuráveis ------------------------------------------------ #
POKEAPI            = "https://pokeapi.co/api/v2"
VERSION_GROUP      = None         # None == aceitar movimentos de qualquer versão
MAX_WORKERS        = 8             # requisições simultâneas
REQUESTS_PER_SEC   = 5             # respeita rate‑limit público (~100/min)
SLEEP_IF_RATELIMIT = 30            # segundos
LOGLEVEL           = logging.INFO

############################################################################
# Utilidades
############################################################################
log = logging.getLogger("complete_moves")
logging.basicConfig(level=LOGLEVEL,
                    format="%(levelname)s: %(message)s")


def normalize_move(move: str) -> str:
    """Remove '-',' ' e põe em UPPER CASE (TACKLE, POWERUPPUNCH)."""
    return move.replace("-", "").replace(" ", "").upper()


def parse_moves_field(value: str) -> List[Tuple[int, str]]:
    """Converte '1,TACKLE,3,GROWL' → [(1,'TACKLE'),(3,'GROWL')]."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [(int(parts[i]), parts[i + 1].upper()) for i in range(0, len(parts), 2)]


def stringify_moves_field(moves: List[Tuple[int, str]]) -> str:
    """Inverte parse_moves_field; mantém ordem ascendente por nível."""
    moves_sorted = sorted(moves, key=lambda x: (x[0], x[1]))
    return ",".join(f"{lvl},{name}" for lvl, name in moves_sorted)


def parse_simple_list(value: str) -> List[str]:
    """'A,B,C' → ['A','B','C'] (em UPPER)"""
    return [
        normalize_move(m)
        for m in value.split(",")
        if m.strip()
    ]


def stringify_simple_list(lst: Iterable[str]) -> str:
    return ",".join(sorted(lst))

# ---------------------------- IGNORE LIST -------------------------------- #
IGNORE_MOVES_FILE = "ignore_moves.txt"

def _load_ignore_moves() -> set[str]:
    """
    Lê ignore_moves.txt (um golpe por linha) e devolve os nomes já
    *normalizados* — sem hífens, sem espaços e em UPPER.
    """
    p = Path(__file__).with_name(IGNORE_MOVES_FILE)
    if not p.exists():
        return set()

    return {
        normalize_move(line.strip())   # <-- strip antes de normalizar
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

IGNORE_MOVES: set[str] = _load_ignore_moves()
# ------------------------------------------------------------------------- #

RETRY_MAP_FILE = "ignore-retry.txt"   

def _load_retry_map() -> dict[str, str]:
    """
    Lê ignore-retry.txt e devolve {INTERNALNAME → nome_para_api}.
    Formato aceito:
        NIDORANF nidoran-f
        FARFETCHD  farfetchd
        WYRDEER = wyrdeer
        HOOPA_UNBOUND hoopao
    • Delimitadores aceitos: espaço(s), tab ou '='.
    • Linhas vazias ou iniciadas com '#' são ignoradas.
    """
    p = Path(__file__).with_name(RETRY_MAP_FILE)
    if not p.exists():
        return {}

    retry: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            left, right = line.split("=", 1)
        else:
            left, right = line.split(maxsplit=1)
        retry[left.strip().upper()] = right.strip()
    return retry

RETRY_MAP: dict[str, str] = _load_retry_map()
# ------------------------------------------------------------------------- #

############################################################################
# Consulta PokéAPI
############################################################################
session = requests.Session()
_last_request_time = 0.0
_rate_tokens = queue.Queue(maxsize=REQUESTS_PER_SEC)
for _ in range(REQUESTS_PER_SEC):
    _rate_tokens.put_nowait(None)


def _rate_limit() -> None:
    """Bloqueia se exceder REQUESTS_PER_SEC."""
    global _last_request_time       
    try:
        _rate_tokens.get_nowait()
    except queue.Empty:
        elapsed = time.time() - _last_request_time
        sleep_for = max(0.0, 1.0 - elapsed)
        time.sleep(sleep_for)
    finally:
        _last_request_time = time.time()
        _rate_tokens.put_nowait(None)


def fetch_pokemon(name: str) -> Optional[dict]:
    """Busca /pokemon/{name}; devolve JSON ou None se não achar."""
    url = f"{POKEAPI}/pokemon/{name.lower()}"
    _rate_limit()
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        log.warning("Hit rate‑limit HTTP 429, sleeping %ss", SLEEP_IF_RATELIMIT)
        time.sleep(SLEEP_IF_RATELIMIT)
        return fetch_pokemon(name)
    r.raise_for_status()
    return r.json()


def extract_moves(poke_json: dict) -> dict[str, set[tuple[int, str]] | set[str]]:
    """
    Retorna um dicionário com três conjuntos:
      • "level" : {(nível, MOVE)}
      • "tutor" : {MOVE, ...}
      • "egg"   : {MOVE, ...}

    Se o mesmo MOVE aparecer em vários version‑groups, mantém **apenas um**
    par (nível, MOVE) — escolhendo o *menor* nível em que o golpe é aprendido.
    """
    level_by_move: dict[str, int] = {}
    tutor_set: set[str] = set()
    egg_set: set[str] = set()

    for entry in poke_json["moves"]:
        move_name = normalize_move(entry["move"]["name"])

        for detail in entry["version_group_details"]:
            # Respeita filtro se VERSION_GROUP não for None
            if VERSION_GROUP and detail["version_group"]["name"] != VERSION_GROUP:
                continue

            method = detail["move_learn_method"]["name"]
            if method == "level-up":
                lvl = detail["level_learned_at"] or 1     
                prev = level_by_move.get(move_name)
                if prev is None or lvl < prev:
                    level_by_move[move_name] = lvl

            elif method == "tutor":
                tutor_set.add(move_name)

            elif method == "egg":
                egg_set.add(move_name)

    level_set = {(lvl, name) for name, lvl in level_by_move.items()}
    return {"level": level_set, "tutor": tutor_set, "egg": egg_set}


############################################################################
# Processamento do arquivo PBS
############################################################################
BLOCK_SEP = re.compile(r"^#-+\s*$", flags=re.M)
FIELD_RE  = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<value>.*)$")


class PBSBlock:
    def __init__(self, text: str):
        self.text   = text
        self.fields = self._parse_fields()

    def _parse_fields(self) -> Dict[str, str]:
        out = {}
        for line in self.text.splitlines():
            m = FIELD_RE.match(line)
            if m:
                out[m.group("key")] = m.group("value")
        return out

    # helpers --------------------------------------------------------------
    @property
    def internal(self) -> str:
        return self.fields.get("InternalName", "")

    def _replace_field(self, key: str, new_value: str) -> None:
        pattern = re.compile(rf"^{key}\s*=.*$", flags=re.M)
        replacement = f"{key} = {new_value}"
        if pattern.search(self.text):
            self.text = pattern.sub(replacement, self.text, count=1)
        else:
            self.text = self.text.rstrip() + f"\n{replacement}\n"

    # --------------------------------------------------------------
    def update_moves(
        self,
        new_level: set[tuple[int, str]],
        new_tutor: set[str],
        new_egg: set[str],
    ) -> dict[str, list[str]]:
        """
        Adiciona movimentos faltantes de acordo com as regras:
          • Não duplicar um golpe já existente em Moves= mesmo que em nível diferente.
          • Ignorar qualquer golpe listado em ignore_moves.txt.
        Retorna um dict com listas dos nomes efetivamente inseridos.
        """
        changelog: dict[str, list[str]] = {"level": [], "tutor": [], "egg": []}

        # ---------------- Level‑up ----------------
        moves_line = self.fields.get("Moves", "")
        current_pairs = parse_moves_field(moves_line) if moves_line else []
        current_names_norm = {normalize_move(n) for _, n in current_pairs}

        missing_level = {
            (lvl, name)
            for (lvl, name) in new_level
            if name not in current_names_norm and name not in IGNORE_MOVES
        }

        if missing_level:
            combined = current_pairs + sorted(missing_level, key=lambda x: (x[0], x[1]))
            self._replace_field("Moves", stringify_moves_field(combined))
            changelog["level"] = [name for _, name in sorted(missing_level)]

        # ---------------- Tutor -------------------
        tutor_line = self.fields.get("TutorMoves", "")
        current_tutor = set(parse_simple_list(tutor_line))

        missing_tutor = {
            m for m in new_tutor if m not in current_tutor and m not in IGNORE_MOVES
        }

        if missing_tutor:
            combined = current_tutor | missing_tutor
            self._replace_field("TutorMoves", stringify_simple_list(combined))
            changelog["tutor"] = sorted(missing_tutor)

        # ---------------- Egg ---------------------
        egg_line = self.fields.get("EggMoves", "")
        current_egg = set(parse_simple_list(egg_line))

        missing_egg = {
            m for m in new_egg if m not in current_egg and m not in IGNORE_MOVES
        }

        if missing_egg:
            combined = current_egg | missing_egg
            self._replace_field("EggMoves", stringify_simple_list(combined))
            changelog["egg"] = sorted(missing_egg)

        return changelog


############################################################################
# pipeline
############################################################################
def process_block(block_text: str) -> tuple[str, Optional[dict[str, list[str]]]]:
    """
    Processa um Pokémon individual.
    Tenta primeiro o InternalName; se falhar e existir entrada em RETRY_MAP,
    faz nova consulta com o alias.
    """
    block = PBSBlock(block_text)
    internal = block.internal
    if not internal:
        return block_text, None

    # --- 1ª tentativa -----------------------------------------------------
    data = fetch_pokemon(internal)

    # --- retry opcional ---------------------------------------------------
    if data is None:
        alias = RETRY_MAP.get(internal.upper())
        if alias:
            data = fetch_pokemon(alias)

    # ---------------------------------------------------------------------
    if data is None:
        return block_text, None  

    movesets = extract_moves(data)
    changes = block.update_moves(movesets["level"], movesets["tutor"], movesets["egg"])
    if any(changes.values()):
        return block.text, changes
    return block.text, {}          


def main(pbs_path: str | Path) -> None:
    pbs_path = Path(pbs_path)
    src = pbs_path.read_text(encoding="utf-8", errors="ignore")

    blocks = BLOCK_SEP.split(src)
    sep_matches = BLOCK_SEP.findall(src) + [""] 

    out_blocks: list[str] = []
    ignore: list[str] = []
    changelog: dict[str, dict[str, list[str]]] = {}

    for blk in tqdm(blocks, desc="Pokémon", unit="pkmn"):
        txt, changes = process_block(blk)
        out_blocks.append(txt)  

        if changes is None:
            ignore.append(PBSBlock(blk).internal)
        elif changes:           
            changelog[PBSBlock(txt).internal] = changes

    # reconstruir arquivo final
    output_text = "".join(sep + blk for sep, blk in zip(sep_matches, out_blocks))
    (pbs_path.parent / "output.txt").write_text(output_text, encoding="utf-8")
    (pbs_path.parent / "ignore.txt").write_text("\n".join(sorted(ignore)), encoding="utf-8")

    with (pbs_path.parent / "changelog.txt").open("w", encoding="utf-8") as fh:
        for pk, changes in sorted(changelog.items()):
            fh.write(f"{pk}:\n")
            for cat in ("level", "tutor", "egg"):
                if changes[cat]:
                    fh.write(f"  + {cat}: {', '.join(changes[cat])}\n")
            fh.write("\n")

    log.info("Pronto! Arquivos gerados em %s", pbs_path.parent.resolve())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python complete_moves.py caminho/para/pokemon.txt")
    main(sys.argv[1])
