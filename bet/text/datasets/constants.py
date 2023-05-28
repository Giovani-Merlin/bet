# * Biencoder
ENT_START_TAG = "[ENT_START]"
ENT_END_TAG = "[ENT_END]"
ENT_TITLE_TAG = "[TITLE]"
# * Zeshel
ZESHEL_WORLDS = [
    "american_football",
    "doctor_who",
    "fallout",
    "final_fantasy",
    "military",
    "pro_wrestling",
    "starwars",
    "world_of_warcraft",
    "coronation_street",
    "muppets",
    "ice_hockey",
    "elder_scrolls",
    "forgotten_realms",
    "lego",
    "star_trek",
    "yugioh",
]
world_to_id = {src: k for k, src in enumerate(ZESHEL_WORLDS)}
