import argparse
import os
import shutil
from pathlib import Path


def normalize_path_separator(path, target_sep='/'):
    """Normalize path separators to target separator (default: forward slash)."""
    return path.replace('\\', target_sep).replace('/', target_sep)


def normalize_prefix(prefix):
    """Normalize prefix to handle both forward and backward slashes."""
    # Normalize to forward slashes for comparison
    prefix = normalize_path_separator(prefix, '/')
    # Ensure prefix ends with separator for proper matching
    if not prefix.endswith('/'):
        prefix += '/'
    return prefix


def main():
    parser = argparse.ArgumentParser(
        description='Convert absolute paths in a text file to relative paths by removing the specified prefix.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Replace paths in file (creates backup)
  python fixpath.py --input train.txt --prefix "E:/KazeRacing25/DesktopTraning"
  
  # Write to different output file
  python fixpath.py --input train.txt --output train_fixed.txt --prefix "E:/KazeRacing25/DesktopTraning"
  
  # No backup
  python fixpath.py --input train.txt --prefix "E:/KazeRacing25/DesktopTraning" --no-backup
        '''
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to the input text file containing absolute paths.')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output file (default: overwrites input file).')
    parser.add_argument('--prefix', type=str, default=None,
                       help='The prefix to remove from each path (default: current directory).')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of original file when overwriting.')
    parser.add_argument('--target-sep', type=str, default='/', choices=['/', '\\'],
                       help='Target path separator for output (default: / for cross-platform).')
    args = parser.parse_args()

    # Set default prefix to current directory if not provided
    if args.prefix is None:
        args.prefix = os.getcwd()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Determine output file
    output_file = args.output if args.output else args.input
    create_backup = not args.no_backup and (output_file == args.input)

    # Normalize prefix for comparison (handle both / and \)
    normalized_prefix = normalize_prefix(args.prefix)

    # Read input file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1

    # Process lines
    new_lines = []
    replaced_count = 0
    skipped_count = 0
    for line in lines:
        stripped_line = line.strip()

        # Normalize the line for comparison (handle both / and \)
        normalized_line = normalize_path_separator(stripped_line, '/')

        # Check if line starts with the normalized prefix
        if normalized_line.startswith(normalized_prefix):
            # Remove prefix
            relative_path = normalized_line[len(normalized_prefix):]
            # Remove leading separator if present
            if relative_path.startswith('/'):
                relative_path = relative_path[1:]

            # Apply target separator and replace with '.'
            relative_path = f"./{normalize_path_separator(relative_path, args.target_sep)}"

            # Preserve original line ending
            line_ending = line[len(stripped_line):] if len(line) > len(stripped_line) else '\n'
            new_lines.append(relative_path + line_ending)
            replaced_count += 1
        else:
            # Keep original line unchanged
            new_lines.append(line)
            if stripped_line:  # Count non-empty lines that weren't replaced
                skipped_count += 1

    # Create backup if needed
    if create_backup:
        backup_file = args.input + '.bak'
        try:
            shutil.copy2(args.input, backup_file)
            print(f"Backup created: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            response = input("Continue without backup? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1

    # Write output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1

    # Print summary
    print(f"\n{'='*60}")
    print("Path Conversion Summary")
    print(f"{'='*60}")
    print(f"Input file:  {args.input}")
    print(f"Output file: {output_file}")
    print(f"Prefix removed: {args.prefix}")
    print(f"Total lines processed: {len(lines)}")
    print(f"Paths replaced: {replaced_count}")
    print(f"Lines unchanged: {skipped_count}")
    if create_backup:
        print(f"Backup saved: {backup_file}")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    exit(main())
