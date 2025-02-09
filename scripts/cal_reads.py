import pysam
import sys

def calculate_alignment_statistics(bam_file):
    bam = pysam.AlignmentFile(bam_file, 'rb', threads=10)

    total_insertions = 0
    total_deletions = 0
    total_mismatches = 0
    total_matches = 0
    total_alignment_length = 0
    total_read_length = 0

    for read in bam:
        if read.is_unmapped:
            continue

        try:
            nm = read.get_tag('NM')
        except KeyError:
            # NM tag not present
            continue

        # Get counts from CIGAR
        cigartuples = read.cigartuples  # list of tuples (operation, length)
        if cigartuples is None:
            continue

        insertions = 0
        deletions = 0
        matches_mismatches = 0

        for (op, length) in cigartuples:
            if op == 0:  # M: alignment match (can be a match or mismatch)
                matches_mismatches += length
            elif op == 1:  # I: insertion to the reference
                insertions += length
            elif op == 2:  # D: deletion from the reference
                deletions += length

        mismatches = nm - insertions - deletions
        matches = matches_mismatches - mismatches

        alignment_length = matches_mismatches + insertions + deletions

        read_length = read.query_length  # Total read length (includes soft-clipped bases)

        total_insertions += insertions
        total_deletions += deletions
        total_mismatches += mismatches
        total_matches += matches
        total_alignment_length += alignment_length
        total_read_length += read_length

    bam.close()

    total_errors = total_insertions + total_deletions + total_mismatches

    if total_alignment_length > 0:
        error_rate = total_errors / total_alignment_length
        mismatch_rate = total_mismatches / total_alignment_length
        insertion_rate = total_insertions / total_alignment_length
        deletion_rate = total_deletions / total_alignment_length
        identity_rate = total_matches / total_alignment_length
    else:
        error_rate = mismatch_rate = insertion_rate = deletion_rate = identity_rate = 0

    if total_read_length > 0:
        relative_length = total_alignment_length / total_read_length
    else:
        relative_length = 0

    print(f"Total matches: {total_matches}")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Total insertions: {total_insertions}")
    print(f"Total deletions: {total_deletions}")
    print(f"Total alignment length: {total_alignment_length}")
    print(f"Total read length: {total_read_length}")
    print(f"Error rate: {error_rate:.6f}")
    print(f"Mismatch rate: {mismatch_rate:.6f}")
    print(f"Insertion rate: {insertion_rate:.6f}")
    print(f"Deletion rate: {deletion_rate:.6f}")
    print(f"Identity rate: {identity_rate:.6f}")
    print(f"Relative length: {relative_length:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input.bam")
        sys.exit(1)
    bam_file = sys.argv[1]
    calculate_alignment_statistics(bam_file)
