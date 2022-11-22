def valid_string(string: str) -> bool:
    """
    Checks if input is a valid string. 
    Returns false in case of empty, null, or noisy string.
    """
    if isinstance(string, str):
        if all(
            [
                len(string) > 0,
                string != "nan",
                string != ".nan",
                string != "",
                string != " ",
                "Unnamed" not in string,
            ]
        ):
            return True
    return False