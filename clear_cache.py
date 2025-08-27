#!/usr/bin/env python3
"""
Utility script to clear TextGrad cache
"""

import os
import shutil
import argparse
import platformdirs


def get_cache_location():
    """Get the TextGrad cache directory location"""
    return platformdirs.user_cache_dir("textgrad")


def list_cache_contents():
    """List all cache contents"""
    cache_dir = get_cache_location()
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    print(f"\nCache location: {cache_dir}")
    print("\nCache contents:")
    print("-" * 50)
    
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(cache_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        
        # Calculate directory size
        dir_size = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        total_size += dir_size
        
        # Print directory info
        dir_name = os.path.basename(root)
        if dir_name:
            size_mb = dir_size / (1024 * 1024)
            print(f'{indent}{dir_name}/ ({size_mb:.2f} MB)')
        
        # Print files in directory
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f'{sub_indent}{file} ({file_size:.2f} KB)')
        
        if len(files) > 5:
            print(f'{sub_indent}... and {len(files) - 5} more files')
    
    print("-" * 50)
    print(f"Total cache size: {total_size / (1024 * 1024):.2f} MB")


def clear_cache(engine=None, confirm=True):
    """
    Clear TextGrad cache
    
    Args:
        engine: Specific engine cache to clear (e.g., 'gpt-4o', 'gpt-3.5-turbo-0125')
                If None, clears entire cache
        confirm: Whether to ask for confirmation before clearing
    """
    cache_dir = get_cache_location()
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return
    
    if engine:
        # Clear specific engine cache
        engine_cache_patterns = [
            f"cache_openai_{engine}.db",  # OpenAI engines
            f"cache_anthropic_{engine}.db",  # Anthropic engines
            f"cache_{engine}.db",  # Generic pattern
            engine  # Direct directory name
        ]
        
        cleared = False
        for pattern in engine_cache_patterns:
            cache_path = os.path.join(cache_dir, pattern)
            if os.path.exists(cache_path):
                if confirm:
                    response = input(f"Clear cache for {engine} at {cache_path}? (y/n): ")
                    if response.lower() != 'y':
                        print("Skipped.")
                        continue
                
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
                print(f"✓ Cleared cache for {engine}")
                cleared = True
        
        if not cleared:
            print(f"No cache found for engine: {engine}")
    else:
        # Clear entire cache
        if confirm:
            response = input(f"Clear entire cache at {cache_dir}? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        shutil.rmtree(cache_dir)
        print(f"✓ Cleared entire cache directory: {cache_dir}")


def clear_specific_engines(engines):
    """Clear cache for specific engines"""
    for engine in engines:
        clear_cache(engine=engine, confirm=False)


def main():
    parser = argparse.ArgumentParser(description="Manage TextGrad cache")
    # parser.add_argument("action", choices=["list", "clear", "clear-all"], 
    #                    help="Action to perform")
    # parser.add_argument("--engine", type=str, 
    #                    help="Specific engine to clear (e.g., gpt-4o)")
    # parser.add_argument("--no-confirm", action="store_true",
    #                    help="Don't ask for confirmation")
    
    args = parser.parse_args()
    
    # if args.action == "list":
    #     list_cache_contents()
    # elif args.action == "clear":
    #     if args.engine:
    #         clear_cache(engine=args.engine, confirm=not args.no_confirm)
    #     else:
    #         print("Please specify an engine with --engine or use 'clear-all'")
    # elif args.action == "clear-all":
    clear_cache(engine=None, confirm=True)


if __name__ == "__main__":
    main()