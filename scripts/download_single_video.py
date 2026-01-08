import json
import os
import subprocess
import argparse

# --- CONFIGURAZIONE ---
TARGET_ID = "9_12" # Default se non specificato nel terminale
LINKS_JSON_PATH = "scripts/downloader_repo/metadata/download_links.json"
OUTPUT_DIR = "data/raw_videos"
# ----------------------

def main():

    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Scarica un video specifico dal dataset.")
    
    parser.add_argument("video_id", type=str, nargs='?', default=None, 
                        help=f"L'ID del video da scaricare (Default: {TARGET_ID})")
    
    args = parser.parse_args()

    if args.video_id is not None:
        target_id = args.video_id
        print(f"ℹ️  Input rilevato. Uso ID: {target_id}")
    else:
        target_id = TARGET_ID
        print(f"ℹ️  Nessun argomento fornito. Uso ID di default: {target_id}")
    # ---------------------------------------------

    # 1. Cartella di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Controllo esistenza file JSON
    if not os.path.exists(LINKS_JSON_PATH):
        print(f"ERRORE: File link non trovato in: {LINKS_JSON_PATH}")
        return

    # 3. Lettura database link
    with open(LINKS_JSON_PATH, "r") as f:
        links_db = json.load(f)

    # 4. Ricerca e Download
    if target_id in links_db:
        video_data = links_db[target_id]
        
        # Uso solo gopro_360p per dimensioni ridotte
        url = video_data.get("gopro_360p")
        
        if url:
            print(f"URL Trovato per {target_id} (360p): {url}")
            
            output_filename = os.path.join(OUTPUT_DIR, f"{target_id}.mp4")
            
            # Controllo per evitare la sovrascrittura
            if os.path.exists(output_filename):
                print(f"⚠️  Il file {output_filename} esiste già")
                return

            print(f"⬇️  Avvio download in {OUTPUT_DIR}...")
            try:
                subprocess.run(["wget", "-O", output_filename, url], check=True)
                print(f"\n✅ SUCCESSO! Video scaricato: {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ ERRORE durante il download: {e}")
        else:
            print(f"❌ La chiave 'gopro_360p' non esiste per il video {target_id}")
            print(f"Chiavi disponibili: {list(video_data.keys())}")
    else:
        print(f"❌ ID {target_id} non trovato nel file JSON.")

if __name__ == "__main__":
    main()