import sys
import json
import gzip
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT
from stavki.data.processors.normalize import TeamMapper

def main():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    print(f"Found {len(files)} raw fixture files.")
    
    mapper = TeamMapper.get_instance()
    
    # Map canonical_name -> sportmonks id
    canon_to_id = {}
    
    for fpath in tqdm(files, desc="Parsing..."):
        try:
            with gzip.open(fpath, 'rt', encoding='UTF-8') as f:
                data = json.load(f)
                
            for p in data.get('participants', []):
                pid = p.get('id')
                pname = p.get('name')
                
                if pid and pname:
                    canon = mapper.map_name(pname)
                    if not canon:
                        canon = pname
                        
                    canon_to_id[canon.lower().strip()] = int(pid)
                    
        except Exception:
            continue
            
    out_path = PROJECT_ROOT / "data" / "mapping" / "canonical_to_id.json"
    with open(out_path, "w") as f:
        json.dump(canon_to_id, f, indent=2)
        
    print(f"Successfully generated {len(canon_to_id)} canonical mappings.")
    
if __name__ == "__main__":
    main()
