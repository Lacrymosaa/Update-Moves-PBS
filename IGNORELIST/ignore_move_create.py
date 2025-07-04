"""
Usage
----
python ignore_move_create.py path/to/moves.txt
"""
from __future__ import annotations

import csv
import sys
import logging
from pathlib import Path
from typing import List, Set

import requests

# ------------------------------------------------------------------------- #
# Configs
# ------------------------------------------------------------------------- #
POKEAPI     = "https://pokeapi.co/api/v2"
OUT_FILE    = "ignore_moves.txt"
REQUEST_TIMEOUT = 20
LOGLEVEL    = logging.INFO
# ------------------------------------------------------------------------- #

logging.basicConfig(format="%(levelname)s: %(message)s", level=LOGLEVEL)
log = logging.getLogger("build_ignore_moves")


# --------------------------- utilidades ---------------------------------- #
def normalize_move(name: str) -> str:
    """Remove hífens e espaços; caixa‑alta (ex.: 'aqua-cutter' → 'AQUACUTTER')."""
    return name.replace("-", "").replace(" ", "").upper()


def title_from_api(name: str) -> str:
    """‘aqua-cutter’ → ‘Aqua Cutter’ (para gravar no ignore_moves.txt)."""
    return name.replace("-", " ").title()


# ------------------- carregar moves do fangame -------------------------- #
def load_pbs_moves(path: Path) -> Set[str]:
    """Lê moves.txt (PBS) e devolve o conjunto de nomes *normalizados*."""
    moves: Set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:  
                continue
            internal = row[1].strip()
            if internal:
                moves.add(normalize_move(internal))
    log.info("Moves no fangame: %s", len(moves))
    return moves


# ------------------------- PokeAPI -------------------------------------- #
def fetch_all_moves() -> List[str]:
    """
    Pega a lista completa de golpes na PokéAPI.
    Usa `limit=10000` para evitar paginação.
    """
    url = f"{POKEAPI}/move?limit=10000"
    log.info("Consultando %s ...", url)
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    names = [item["name"] for item in data["results"]]
    log.info("Moves na PokéAPI: %s", len(names))
    return names


# --------------------------- main --------------------------------------- #
def main(moves_pbs_path: str | Path) -> None:
    """
    1. Carrega o moves.txt do fangame.
    2. Baixa a lista completa de golpes da PokéAPI.
    3. Seleciona apenas os golpes que não existem no fangame.
    4. Remove duplicatas “Physical/Special” (mantém só um, sem espaço).
    5. Salva o resultado em ignore_moves.txt.
    """
    moves_pbs_path = Path(moves_pbs_path)
    if not moves_pbs_path.exists():
        sys.exit(f"Arquivo não encontrado: {moves_pbs_path}")

    # ----- 1) Moves do fangame -----------------
    moves_fangame = load_pbs_moves(moves_pbs_path)

    # ----- 2) Moves da PokéAPI -----------------
    api_moves_raw = fetch_all_moves()

    # ----- 3) Diferença (faltantes) ------------
    missing_raw: list[str] = [
        title_from_api(api_name)              
        for api_name in api_moves_raw
        if normalize_move(api_name) not in moves_fangame
    ]

    # ----- 4) Deduplicar Physical/Special ------
    unique_keys: set[str] = set()
    deduped: list[str] = []

    for name in missing_raw:
        key = name.replace(" ", "").upper()   
        if key in unique_keys:
            continue                          
        unique_keys.add(key)
        deduped.append(name.replace(" ", ""))  

    deduped.sort()

    # ----- 5) Escrever ignore_moves.txt --------
    out_path = moves_pbs_path.parent / OUT_FILE
    out_path.write_text("\n".join(deduped), encoding="utf-8")
    log.info("Gerado %s (%s golpes ausentes).", out_path, len(deduped))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python build_ignore_moves.py caminho/para/moves.txt")
    main(sys.argv[1])
