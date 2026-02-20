#!/usr/bin/env python3
"""
A fuzzy git rerere implementation that allows for approximate conflict resolution matching
based on configurable context similarity thresholds.
"""

import os
import sys
import difflib
import subprocess
from pathlib import Path
import hashlib
import json
import random
import string
import shutil
import datetime
import urllib.parse

START_MARKER = '<<<<<<<'
MID_MARKER = '======='
DIFF_3_MID_MARKER = '|||||||'
END_MARKER = '>>>>>>>'

class Rerereric:
    def __init__(self, git_dir=None):
        self.git_dir = git_dir if git_dir else self._get_git_dir()
        self.rerere_dir = Path(self.git_dir) / "rerereric"
        self.rerere_dir.mkdir(exist_ok=True)
        (self.rerere_dir / 'pre').mkdir(exist_ok=True)
        (self.rerere_dir / 'res').mkdir(exist_ok=True)

    def _hash_record(self, record):
        """Create a hash of the entire conflict record including context."""
        content = (
            record["conflict"] +
            record.get("before_context", "") +
            record.get("after_context", "") +
            str(record.get("start_line", "")) +
            str(record.get("end_line", "")) +
            record.get("resolved_at", "")
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_git_dir(self):
        """Get the .git directory for the current repository."""
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _hash_conflict(self, conflict_text):
        """Create a hash of the conflict content."""
        return hashlib.sha256(conflict_text.encode()).hexdigest()[:16]

    def _extract_conflict_markers(self, file_path, context_lines):
        """Extract conflict markers and their context from a file."""
        conflicts = []
        in_conflict = False
        
        with open(file_path) as f:
            lines = self._normalize_conflict_markers(f.read().split('\n'))
            
        for i, line in enumerate(lines):
            if line == START_MARKER:
                in_conflict = True
                conflict_start_line = i
            elif line == END_MARKER:
                if not in_conflict:
                    raise ValueError(f"Unexpected conflict end marker at line {i}")

                # the conflict itself, including markers
                conflict_lines = lines[conflict_start_line:i + 1]

                # find context lines before the conflict (but after the previous one if any and excluding empty lines)
                before_lines = []
                before_idx = conflict_start_line - 1
                while len(before_lines) < context_lines and before_idx >= 0:
                    if lines[before_idx] == END_MARKER:
                        break
                    if lines[before_idx].strip():
                        before_lines.insert(0, lines[before_idx])
                    before_idx -= 1

                # do the same for lines after the conflict
                after_lines = []
                after_idx = i + 1
                while len(after_lines) < context_lines and after_idx < len(lines):
                    if lines[after_idx] == START_MARKER:
                        break
                    if lines[after_idx].strip():
                        after_lines.append(lines[after_idx])                    
                    after_idx += 1

                conflicts.append({
                    'conflict': '\n'.join(conflict_lines),
                    'before_context': '\n'.join(before_lines),
                    'after_context': '\n'.join(after_lines),
                    'start_line': conflict_start_line,
                    'end_line': i,
                    'file_path': str(file_path)
                })

                in_conflict = False
                    
        return conflicts

    def _calculate_context_similarity(self, context1_before, context1_after, context2_before, context2_after):
        """Calculate the similarity between two sets of context."""
        before_ratio = difflib.SequenceMatcher(None, context1_before, context2_before).ratio()
        after_ratio = difflib.SequenceMatcher(None, context1_after, context2_after).ratio()
        return (before_ratio + after_ratio) / 2

    def _find_similar_resolution(self, conflict_info, similarity_threshold, context_lines):
        """Find a similar conflict resolution from the stored records."""
        matches = []
        current_file = conflict_info['file_path']
        current_conflict = conflict_info['conflict']
        current_line = conflict_info['start_line']
        
        # only look at files with matching conflict hash
        conflict_hash = self._hash_conflict(current_conflict)
        for record_file in (self.rerere_dir / 'res').glob(f"{conflict_hash}_*.json"):
            with open(record_file) as f:
                record = json.load(f)
                
                # first check for exact conflict match
                if record["conflict"] != current_conflict:
                    continue
                
                record_before_context = '\n'.join(record.get('before_context', '').split('\n')[-context_lines:])
                record_after_context = '\n'.join(record.get('after_context', '').split('\n')[:context_lines])

                # calculate context similarity
                context_similarity = self._calculate_context_similarity(
                    conflict_info['before_context'],
                    conflict_info['after_context'],
                    # get first n context lines
                    record_before_context,
                    record_after_context
                )
                
                # only consider matches with sufficient context similarity
                if context_similarity >= similarity_threshold:
                    matches.append({
                        'resolution': record['resolution'],
                        'context_similarity': context_similarity,
                        'same_file': record['file_path'] == current_file,
                        'line_distance': abs(record.get('start_line', 0) - current_line),
                        'file_path': record['file_path'],
                        'start_line': record['start_line'],
                        'end_line': record['end_line'],
                        'resolved_at': record['resolved_at']
                    })

        if not matches:
            return None, 0

        # sort matches by priority:
        # 1. same file
        # 2. context similarity
        # 3. line number proximity
        # 4. resolution timestamp
        matches.sort(key=lambda x: (
            x['same_file'],
            x['context_similarity'],
            -x['line_distance'],
            x["resolved_at"]
        ), reverse=True)

        best_match = matches[0]
        return best_match['resolution'], best_match['context_similarity']

    def get_pre_path_from_file_path(self, file_path):
        """Get the path to the pre-resolution state for a file."""
        file_id = urllib.parse.quote(str(Path(file_path)), safe='')
        return self.rerere_dir / 'pre' / f"{file_id}.pre"

    def get_file_path_from_pre_path(self, pre_path):
        """Get the path to the file from the pre-resolution state."""
        if self.should_get_file_path_from_pre_path_old(pre_path):
            return self.get_file_path_from_pre_path_old(pre_path)

        file_id = pre_path.stem
        return urllib.parse.unquote(file_id.replace('.pre', '')).replace(str(self.rerere_dir) + '/', '')

    def should_get_file_path_from_pre_path_old(self, pre_path):
        if '%2F' in pre_path.stem:
            return False

        # only use old method if we have high confidence it was encoded with __'s (no slashes encoded
        # as %2F, no __ at the start of the stem, and at least one __ in the stem)
        if '__' in pre_path.stem and not pre_path.stem.startswith('__'):
            return True

        return False

    def get_file_path_from_pre_path_old(self, pre_path):
        """Get the path to the file from the pre-resolution state, old version where we used to replace slashes with __."""
        file_id = pre_path.stem
        return file_id.replace('__', '/').replace('.pre', '').replace(str(self.rerere_dir) + '/', '')

    def mark_conflicts(self, file_paths):
        """Save the entire file state before conflict resolution."""
        for file_path in file_paths:
            pre_path = self.get_pre_path_from_file_path(file_path)
            print(f"Saving pre-resolution state to {pre_path}")
            shutil.copyfile(file_path, pre_path)
        return True

    def split_conflict(self, conflict):
        """Split a conflict into its components."""
        parts = conflict.split('\n')
        start = parts.index(START_MARKER)
        mid = parts.index(MID_MARKER)
        end = parts.index(END_MARKER)
        
        if DIFF_3_MID_MARKER in parts:
            diff3_mid = parts.index(DIFF_3_MID_MARKER)
            return [
                parts[start + 1:diff3_mid],
                parts[diff3_mid + 1:mid],
                parts[mid + 1:end]
            ]
        else:
            return [
                parts[start + 1:mid],
                parts[mid + 1:end]
            ]

    def save_resolutions(self, context_lines=2):
        """Save resolutions after conflicts have been manually resolved."""
        # find matching pre-resolution state
        for pre_file in (self.rerere_dir / 'pre').glob('*.pre'):
            file_path = self.get_file_path_from_pre_path(pre_file)

            # load and check pre-resolution content
            with open(pre_file) as f:
                pre_content = f.read().split('\n')

            # extract conflicts from pre-resolution state
            conflicts = self._extract_conflict_markers(pre_file, context_lines)
            if not conflicts:
                print(f"No conflicts found in pre-resolution state for {file_path} with content {pre_content}")
                pre_file.unlink()  # clean up pre file since it has no conflicts
                continue

            # load post-resolution content
            with open(file_path) as f:
                post_content = f.read().split('\n')

            # process each conflict, tracking line offsets
            resolutions = []
            line_offset = 0

            for conflict in conflicts:
                parts = self.split_conflict(conflict["conflict"])
                parts.sort(key=len, reverse=True)

                # where the pre-resolution state starts and ends
                pre_start = conflict["start_line"]
                pre_end = conflict["end_line"]

                # where the post-resolution state starts
                post_start = pre_start + line_offset

                if len(conflicts) == 1:
                    # if there's only one conflict, our job is easy; we know the difference between the line
                    # lengths of the files and the difference between the conflict and resolution lengths must
                    # be the same
                    post_end = pre_end + len(post_content) - len(pre_content)

                else:
                    for part in parts:
                        if '\n'.join(part) == '\n'.join(post_content[post_start:post_start + len(part)]):
                            # if a chunk of content matches one of the conflict parts exactly, assume that is the resolution
                            post_end = post_start + len(part) - 1 # -1 because in's inclusive
                            break
                    else:
                        # otherwise we'll try something different - try to find N lines after the conflict text that match
                        # N lines in the post-resolution content; the lines preceeding those N lines in the post-resolution
                        # content will be the resolution
                        
                        # post_end is the proposed last line of the resolution, which we will increment in the while loop below
                        post_end = post_start

                        # pre_line and post_line are the lines that we are comparing for equality
                        pre_line = pre_end + 1
                        post_line = post_end + 1

                        matches = 0

                        matched = False

                        # look for meaningful matching content after the conflict
                        # try to match next N non-empty lines
                        REQUIRED_MATCHING_LINES = 3

                        while post_line < len(post_content) and pre_line < len(pre_content):
                            # once we hit the next conflict marker, we're obviously past the resolution, so
                            # let's use whatever we have stored in post_end
                            if pre_content[pre_line] == START_MARKER:
                                matched = matches >= 1
                                break

                            if post_content[post_line] == pre_content[pre_line]:
                                # only actually count the match if it's a non-empty line
                                if pre_content[pre_line].strip():
                                    matches += 1

                                # we got a match, so if we loop again we'll check the next line
                                pre_line += 1
                                post_line += 1

                            else:
                                # we got a mismatch, so we'll try the next line in the post-resolution content
                                post_end += 1

                                # reset the match count and line offsets
                                matches = 0
                                pre_line = pre_end + 1
                                post_line = post_end + 1

                            if matches == REQUIRED_MATCHING_LINES:
                                break
                                
                        else:
                            # consider it a match if we hit the content end while matching
                            matched = matches >= 1

                        matched = matched or matches >= REQUIRED_MATCHING_LINES

                        print(pre_line, post_line, matches)

                        resolution_length = post_end - post_start + 1
                        largest_conflict_part = max(len(part) for part in parts)

                        if not matched:
                            # if we didn't find enough matching lines, assume that we didn't find a real match
                            continue

                resolution = '\n'.join(post_content[post_start:post_end + 1])                

                # update line offset for next conflict
                conflict_line_count = pre_end - pre_start + 1
                resolution_line_count = post_end - post_start + 1
                line_offset += resolution_line_count - conflict_line_count

                resolutions.append({
                    "conflict": conflict["conflict"],
                    "resolution": resolution,
                    "before_context": conflict["before_context"],
                    "after_context": conflict["after_context"],
                    "start_line": conflict["start_line"],
                    "end_line": conflict["end_line"],
                    "resolved_at": datetime.datetime.now().isoformat()
                })

            # save each resolution separately
            for resolution in resolutions:
                # create unique hash for this specific resolution
                conflict_hash = self._hash_conflict(resolution["conflict"])
                record_hash = self._hash_record(resolution)
                record_path = self.rerere_dir / 'res' / f"{conflict_hash}_{record_hash}.json"

                record = {
                    "file_path": str(file_path),
                    "conflict": resolution["conflict"],
                    "resolution": resolution["resolution"],
                    "before_context": resolution["before_context"],
                    "after_context": resolution["after_context"],
                    "start_line": resolution["start_line"],
                    "end_line": resolution["end_line"],
                    "resolved_at": resolution["resolved_at"]
                }
                with open(record_path, 'w') as f:
                    json.dump(record, f, indent=2)

            # clean up temporary files
            pre_file.unlink()

    def _apply_resolution(self, file_path, conflict_info, resolution):
        """Apply a resolution to a specific conflict in a file."""
        with open(file_path, 'r', newline='') as f:
            raw = f.read()

        line_ending = '\r\n' if '\r\n' in raw else '\n'
        content = raw.split(line_ending)

        # replace the conflict with the resolution
        content[conflict_info['start_line']:conflict_info['end_line'] + 1] = resolution.split('\n')

        with open(file_path, 'w', newline='') as f:
            f.write(line_ending.join(content))

    def reapply_resolutions(self, file_paths, similarity_threshold=0.8, context_lines=2):
        """Try to resolve conflicts in a file using stored resolutions."""
        resolved = []
        unresolved = []

        for file_path in file_paths:
            conflicts = self._extract_conflict_markers(file_path, context_lines)

            # process conflicts from bottom to top to maintain line numbers
            for conflict_info in reversed(conflicts):
                resolution, confidence = self._find_similar_resolution(conflict_info, similarity_threshold, context_lines)

                if resolution is not None:
                    print(f"Found similar resolution with {confidence:.2%} confidence")
                    self._apply_resolution(file_path, conflict_info, resolution)
                    resolved.append(file_path)
                    print(f"Applied resolution from {conflict_info['file_path']} "
                        f"at line {conflict_info['start_line']}")
                else:
                    unresolved.append(file_path)
                    print(f"No similar resolution found for conflict in {file_path} "
                        f"at line {conflict_info['start_line']}")

        return resolved, unresolved

    def _normalize_conflict_markers(self, lines):
        """Normalize conflict markers in a list of lines."""
        normalized = []
        for line in lines:
            for marker in [START_MARKER, MID_MARKER, DIFF_3_MID_MARKER, END_MARKER]:
                if line.startswith(marker):
                    line = marker
                    break
            normalized.append(line)
        return normalized

