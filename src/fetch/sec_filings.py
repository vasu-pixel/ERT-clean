from sec_edgar_downloader import Downloader
import os

def download_latest_10k(ticker: str, save_dir="data/sec_filings"):
    print(f"🔍 Attempting to download 10-K for {ticker}...")
    print(f"📁 Save directory: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # The Downloader requires an email address for SEC compliance
        dl = Downloader(save_dir, email_address="user@example.com")
        
        # Download the latest 10-K filing
        print(f"📥 Downloading 10-K filing for {ticker}...")
        result = dl.get("10-K", ticker)
        print(f"✅ Download completed for {ticker}. Files downloaded: {result}")
        
    except Exception as e:
        print(f"❌ Error downloading filing: {e}")
        return None
    
    # Get the most recent file path - SEC downloader creates files in project root
    # Try both possible locations
    possible_paths = [
        os.path.join(save_dir, "sec-edgar-filings", ticker, "10-K"),  # Expected path
        os.path.join("sec-edgar-filings", ticker, "10-K"),           # Actual path
    ]
    
    ticker_path = None
    for path in possible_paths:
        print(f"🔍 Checking path: {path}")
        if os.path.exists(path):
            ticker_path = path
            print(f"✅ Found directory: {ticker_path}")
            break
    
    if not ticker_path:
        print(f"❌ Directory does not exist: {ticker_path}")
        
        # Let's check what was actually created in the save directory
        print(f"🔍 Checking full directory structure of: {save_dir}")
        if os.path.exists(save_dir):
            # Find all files and show the structure
            all_files = []
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, save_dir)
                    file_size = os.path.getsize(full_path)
                    all_files.append((rel_path, file_size, full_path))
            
            print(f"📂 Found {len(all_files)} files:")
            for rel_path, file_size, full_path in all_files[:10]:  # Show first 10 files
                print(f"   {rel_path} ({file_size} bytes)")
            
            if len(all_files) > 10:
                print(f"   ... and {len(all_files) - 10} more files")
                
            # If we found files, try to return the largest one that looks like a 10-K
            if all_files:
                # Look for files that might be the main 10-K document
                likely_10k_files = [
                    (path, size, full) for path, size, full in all_files 
                    if any(keyword in path.lower() for keyword in ['10-k', 'full-submission', 'filing'])
                ]
                
                if likely_10k_files:
                    # Return the largest likely 10-K file
                    best_file = max(likely_10k_files, key=lambda x: x[1])
                    print(f"✅ Found likely 10-K file: {best_file[0]} ({best_file[1]} bytes)")
                    return best_file[2]
                else:
                    # Return the largest file overall
                    best_file = max(all_files, key=lambda x: x[1])
                    print(f"✅ Using largest available file: {best_file[0]} ({best_file[1]} bytes)")
                    return best_file[2]
        else:
            print(f"❌ Save directory doesn't exist: {save_dir}")
        
        return None
        
    # Find the most recent filing directory
    try:
        subdirs = [d for d in os.listdir(ticker_path) if os.path.isdir(os.path.join(ticker_path, d))]
        print(f"📂 Found subdirectories: {subdirs}")

        if not subdirs:
            print(f"❌ No subdirectories found in {ticker_path}")
            return None

        def sort_key(name: str) -> tuple[int, int]:
            parts = name.split('-')
            if len(parts) >= 3 and parts[1].isdigit():
                try:
                    year = int(parts[1])
                    seq = int(parts[2]) if parts[2].isdigit() else 0
                    return (year, seq)
                except ValueError:
                    pass
            return (0, 0)

        subdirs.sort(key=sort_key, reverse=True)
        most_recent_dir = subdirs[0]
        print(f"📅 Most recent filing directory: {most_recent_dir}")
        
        # Look for the main filing document
        filing_dir = os.path.join(ticker_path, most_recent_dir)
        
        # Try different possible file names
        possible_files = [
            "full-submission.txt",
            "filing-details.html", 
            os.listdir(filing_dir)[0] if os.listdir(filing_dir) else None  # First file in directory
        ]
        
        for filename in possible_files:
            if filename:
                doc_path = os.path.join(filing_dir, filename)
                print(f"📄 Checking for file: {doc_path}")
                
                if os.path.exists(doc_path):
                    file_size = os.path.getsize(doc_path)
                    print(f"✅ Found filing file: {filename}! Size: {file_size} bytes")
                    return doc_path
        
        # If no specific file found, list what's available
        print(f"📂 Files available in {filing_dir}:")
        for item in os.listdir(filing_dir):
            item_path = os.path.join(filing_dir, item)
            if os.path.isfile(item_path):
                file_size = os.path.getsize(item_path)
                print(f"   - {item} ({file_size} bytes)")
        
        # Return the largest file (likely the main document)
        files_in_dir = [f for f in os.listdir(filing_dir) if os.path.isfile(os.path.join(filing_dir, f))]
        if files_in_dir:
            largest_file = max(files_in_dir, key=lambda f: os.path.getsize(os.path.join(filing_dir, f)))
            doc_path = os.path.join(filing_dir, largest_file) 
            print(f"✅ Using largest file: {largest_file}")
            return doc_path
    
    except Exception as e:
        print(f"❌ Error processing directories: {e}")
    
    return None
