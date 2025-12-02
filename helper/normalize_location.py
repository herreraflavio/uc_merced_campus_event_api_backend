import re

LOCATION_MAP: dict[str, str] = {}


def normalize_location(raw_location: str | None) -> str | None:
    """
    1. PROTECTS specific tokens like COB1/COB2 from number stripping.
    2. Removes other numbers (room numbers) and special characters.
    3. Checks if the result matches a known alias.
    4. Returns the Canonical Name if matched.

    Example: "COB2 130" -> Protect "COB2" -> Remove "130" -> "COB2" -> Lookup
    """
    if not raw_location:
        return None

    # 1. Lowercase the input
    text = raw_location.lower()

    # 2. PROTECTION: Handle specific cases where we WANT the number kept.
    #    We replace them with a temporary text-only placeholder so the next step
    #    doesn't delete the number.
    #    Matches: "cob1", "cob 1", "cob-1", "cob2", "cob 2", "cob-2"
    text = re.sub(r'\bcob[\s-]?1\b', '__cob_one__', text)
    text = re.sub(r'\bcob[\s-]?2\b', '__cob_two__', text)

    # 3. Remove all remaining numbers (e.g., the "130" in "COB2 130")
    text = re.sub(r'\d+', '', text)

    # 4. Restore the protected tokens back to "cob1" / "cob2"
    text = text.replace('__cob_one__', 'cob1')
    text = text.replace('__cob_two__', 'cob2')

    # 5. Remove special characters (leaves only letters, numbers, underscores, spaces)
    #    We allow alphanumeric so "cob1" survives.
    text = re.sub(r'[^\w\s]', '', text)

    # 6. Collapse multiple spaces into one and strip edges
    cleaned_text = " ".join(text.split())

    # 7. Check for Exact Match in our map
    if cleaned_text in LOCATION_MAP:
        return LOCATION_MAP[cleaned_text]

    # 8. Fallback: Check if a known alias exists *inside* the string
    sorted_aliases = sorted(LOCATION_MAP.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, cleaned_text):
            return LOCATION_MAP[alias]

    return raw_location.strip()
