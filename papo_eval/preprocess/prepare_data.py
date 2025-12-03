import os
import json
import zipfile
import argparse
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
from pathlib import Path

def download_dataset_split_as_json(repo_id, split_name, output_dir, hf_token=None):
    """Download a specific split from HuggingFace dataset and save as JSON"""
    
    if hf_token:
        login(token=hf_token)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading split '{split_name}' from {repo_id}...")
    
    # Load specific split
    dataset = load_dataset(repo_id, split=split_name)
    
    # Convert to list of dictionaries (ShareGPT format)
    data_list = []
    for item in dataset:
        # Remove the 'id' field if it exists (since it was added during upload)
        if 'id' in item:
            del item['id']
        data_list.append(item)
    
    # Save as JSON
    json_filename = f"{split_name}.json"
    json_path = output_dir / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(data_list)} items to {json_path}")
    return json_path

def download_image_zip(repo_id, zip_filename, output_dir, hf_token=None):
    """Download image ZIP file from HuggingFace dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading {zip_filename} from {repo_id}...")
        
        # Download the ZIP file
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
            token=hf_token
        )
        
        # Copy to output directory
        output_zip_path = output_dir / zip_filename
        import shutil
        shutil.copy2(zip_path, output_zip_path)
        
        print(f"✅ Downloaded ZIP file to {output_zip_path}")
        return output_zip_path
        
    except Exception as e:
        print(f"❌ Error downloading {zip_filename}: {e}")
        return None

def extract_images_from_zip(zip_path, extract_dir, auto_remove=False):
    """Extract images from ZIP file and optionally remove the ZIP"""
    
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"✅ Extracted images to {extract_dir}")
        
        # Remove ZIP file if auto_remove is True
        if auto_remove:
            os.remove(zip_path)
            print(f"🗑️  Removed ZIP file: {zip_path}")
        
        return extract_dir
        
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return None

def download_complete_dataset_split(repo_id, split_name, zip_filename, output_dir, hf_token=None, auto_unzip=False):
    """Download both JSON data and images for a specific dataset split"""
    
    output_dir = Path(output_dir)
    
    # JSON files go to OUTPUT_DIR/papo/
    json_dir = output_dir / "papo"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary directory for ZIP download (will be moved to images if auto_unzip)
    temp_zip_dir = output_dir / "temp" / split_name
    temp_zip_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Processing {split_name} ===")
    
    # Download JSON data to papo folder
    json_path = download_dataset_split_as_json(
        repo_id=repo_id,
        split_name=split_name,
        output_dir=json_dir,
        hf_token=hf_token
    )
    
    # Download image ZIP to temp directory
    zip_path = download_image_zip(
        repo_id=repo_id,
        zip_filename=zip_filename,
        output_dir=temp_zip_dir,
        hf_token=hf_token
    )
    
    # Extract images if auto_unzip is True
    images_dir = None
    if auto_unzip and zip_path:
        # Images go directly to OUTPUT_DIR/images/
        images_dir = output_dir / "images"
        extract_images_from_zip(zip_path, images_dir, auto_remove=True)
        
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_zip_dir.parent)  # Remove temp folder
        except:
            pass
    else:
        # If not auto-unzipping, move ZIP to main output directory
        if zip_path:
            final_zip_path = output_dir / zip_filename
            import shutil
            shutil.move(str(zip_path), str(final_zip_path))
            zip_path = final_zip_path
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_zip_dir.parent)
            except:
                pass
    
    result = {
        'json_path': json_path,
        'zip_path': zip_path if not auto_unzip else None,
        'images_dir': images_dir,
        'json_dir': json_dir,
        'output_dir': output_dir
    }
    
    print(f"✅ Completed processing {split_name}")
    return result

def get_dataset_config():
    """Get the mapping between split names and ZIP filenames"""
    return {
        'AI4Math_MathVerse': 'AI4Math_MathVerse_images.zip',
        'AI4Math_MathVerse_vision_dependent': 'AI4Math_MathVerse_vision_dependent_images.zip',
        'AI4Math_MathVista': 'AI4Math_MathVista_images.zip',
        'BUAADreamer_clevr_count_70k': 'BUAADreamer_clevr_count_70k_images.zip',
        'hiyouga_geometry3k': 'hiyouga_geometry3k_images.zip',
        'lscpku_LogicVista': 'lscpku_LogicVista_images.zip',
        'MMMU_MMMU_Pro': 'MMMU_MMMU_Pro_images.zip',
        'We_Math': 'We-Math_We-Math_images.zip',
        'PAPO_MMK12': 'PAPO_MMK12_test_images.zip',
    }

def main():
    parser = argparse.ArgumentParser(description='Download dataset splits from HuggingFace')
    parser.add_argument('--repo_id', type=str, default='PAPO-Galaxy/PAPO_eval',
                       help='HuggingFace repository ID')
    parser.add_argument('--split_name', type=str, required=True,
                       help='Name of the dataset split to download')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for downloaded data')
    parser.add_argument('--hf_token', type=str,
                       help='HuggingFace token (can also be set via HF_TOKEN env var)')
    parser.add_argument('--auto_unzip', action='store_true',
                       help='Automatically unzip images and remove ZIP files')
    parser.add_argument('--list_splits', action='store_true',
                       help='List all available splits and exit')
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    # Get dataset configuration
    dataset_config = get_dataset_config()
    
    # List available splits if requested
    if args.list_splits:
        print("Available dataset splits:")
        for split_name, zip_filename in dataset_config.items():
            print(f"  - {split_name} → {zip_filename}")
        return
    
    # Validate split name
    if args.split_name not in dataset_config:
        print(f"❌ Error: Split '{args.split_name}' not found.")
        print("Available splits:")
        for split_name in dataset_config.keys():
            print(f"  - {split_name}")
        return
    
    # Get corresponding ZIP filename
    zip_filename = dataset_config[args.split_name]
    
    print(f"Starting download for split: {args.split_name}")
    print(f"Repository: {args.repo_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"ZIP filename: {zip_filename}")
    print(f"Auto unzip: {args.auto_unzip}")
    
    # Download the dataset
    try:
        result = download_complete_dataset_split(
            repo_id=args.repo_id,
            split_name=args.split_name,
            zip_filename=zip_filename,
            output_dir=args.output_dir,
            hf_token=hf_token,
            auto_unzip=args.auto_unzip
        )
        
        print(f"\n🎉 Successfully downloaded {args.split_name}")
        print(f"📁 Main directory: {result['output_dir']}")
        if result['json_path']:
            print(f"📄 JSON file: {result['json_path']}")
        if result['zip_path']:
            print(f"🗜️  ZIP file: {result['zip_path']}")
        if result['images_dir']:
            print(f"🖼️  Images directory: {result['images_dir']}")
        
        # Show the final structure
        print(f"\n📂 Final structure:")
        print(f"   {result['output_dir']}/")
        print(f"   ├── papo/")
        print(f"   │   └── {args.split_name}.json")
        if result['images_dir']:
            print(f"   └── images/")
            print(f"       └── (extracted image folders)")
        elif result['zip_path']:
            print(f"   └── {Path(result['zip_path']).name}")
            
    except Exception as e:
        print(f"❌ Error downloading {args.split_name}: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())