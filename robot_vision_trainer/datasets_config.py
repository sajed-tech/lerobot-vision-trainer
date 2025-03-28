"""
Configuration file containing LeRobot datasets organized by task types.
"""

# Define task categories and associated datasets
LEROBOT_DATASETS = {
    "manipulation": {
        "pick_and_place": [
            "lerobot/metaworld_mt50",
            "lerobot/ucsd_pick_and_place_dataset",
            "lerobot/utokyo_xarm_pick_and_place",
            "lerobot/koch_pick_place_5_lego",
            "lerobot/koch_pick_place_1_lego",
            "lerobot/iamlab_cmu_pickup_insert"
        ],
        "push": [
            "lerobot/metaworld_mt50_push_v2_image",
            "lerobot/pusht",
            "lerobot/pusht_image",
            "lerobot/pusht_keypoints",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/xarm_push_medium",
            "lerobot/xarm_push_medium_image",
            "lerobot/xarm_push_medium_replay",
            "lerobot/xarm_push_medium_replay_image"
        ],
        "lift": [
            "lerobot/xarm_lift_medium",
            "lerobot/xarm_lift_medium_image",
            "lerobot/xarm_lift_medium_replay",
            "lerobot/xarm_lift_medium_replay_image",
            "lerobot/aloha_static_fork_pick_up"
        ],
        "pour": [
            "lerobot/dlr_sara_pour",
            "lerobot/aloha_static_coffee",
            "lerobot/aloha_static_coffee_new"
        ],
        "insertion": [
            "lerobot/aloha_sim_insertion_scripted",
            "lerobot/aloha_sim_insertion_scripted_image",
            "lerobot/aloha_sim_insertion_human",
            "lerobot/aloha_sim_insertion_human_image"
        ]
    },
    "household": {
        "kitchen_tasks": [
            "lerobot/stanford_robocook",
            "lerobot/ucsd_kitchen_dataset",
            "lerobot/taco_play",
            "lerobot/aloha_mobile_wash_pan",
            "lerobot/aloha_mobile_shrimp"
        ],
        "cleaning": [
            "lerobot/aloha_mobile_wipe_wine",
            "lerobot/aloha_static_towel"
        ],
        "furniture": [
            "lerobot/aloha_mobile_chair",
            "lerobot/aloha_mobile_cabinet",
            "lerobot/utokyo_pr2_opening_fridge"
        ],
        "clothing": [
            "lerobot/unitreeh1_fold_clothes",
            "lerobot/usc_cloth_sim"
        ]
    },
    "navigation": {
        "indoor_navigation": [
            "lerobot/berkeley_gnm_cory_hall",
            "lerobot/berkeley_gnm_sac_son",
            "lerobot/berkeley_gnm_recon",
            "lerobot/aloha_mobile_elevator"
        ],
        "warehouse": [
            "lerobot/unitreeh1_warehouse"
        ]
    },
    "bimanual": {
        "bimanual_manipulation": [
            "lerobot/utokyo_xarm_bimanual",
            "lerobot/aloha_static_cups_open",
            "lerobot/aloha_static_candy",
            "lerobot/aloha_static_screw_driver",
            "lerobot/aloha_static_vinh_cup",
            "lerobot/aloha_static_vinh_cup_left",
            "lerobot/aloha_static_ziploc_slide",
            "lerobot/aloha_static_thread_velcro"
        ],
        "transfer": [
            "lerobot/aloha_sim_transfer_cube_scripted",
            "lerobot/aloha_sim_transfer_cube_scripted_image",
            "lerobot/aloha_sim_transfer_cube_human",
            "lerobot/aloha_sim_transfer_cube_human_image"
        ]
    },
    "special_objects": {
        "cables": [
            "lerobot/berkeley_cable_routing",
            "lerobot/conq_hose_manipulation"
        ],
        "utensils": [
            "lerobot/aloha_static_pro_pencil",
            "lerobot/aloha_static_tape",
            "lerobot/aloha_static_battery"
        ],
        "small_objects": [
            "lerobot/aloha_static_pingpong_test"
        ]
    },
    "diverse_data": {
        "multimodal": [
            "lerobot/stanford_kuka_multimodal_dataset",
            "lerobot/cmu_play_fusion"
        ],
        "general_purpose": [
            "lerobot/libero_10_image",
            "lerobot/libero_object_image",
            "lerobot/libero_goal_image",
            "lerobot/libero_spatial_image",
            "lerobot/roboturk",
            "lerobot/berkeley_rpt",
            "lerobot/droid_100",
            "lerobot/fmb"
        ],
        "exploration": [
            "lerobot/nyu_franka_play_dataset",
            "lerobot/jaco_play",
            "lerobot/cmu_franka_exploration_dataset"
        ]
    }
}

# Define object-action mapping to dataset categories
OBJECT_ACTION_MAPPING = {
    # Common household objects
    "cup": {
        "pick up": ["manipulation/pick_and_place", "manipulation/lift"],
        "pour liquid": ["manipulation/pour"],
        "place": ["manipulation/pick_and_place"]
    },
    "bottle": {
        "pick up": ["manipulation/pick_and_place", "manipulation/lift"],
        "pour": ["manipulation/pour"],
        "place": ["manipulation/pick_and_place"]
    },
    "bowl": {
        "pick up": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"],
        "fill": ["manipulation/pour", "household/kitchen_tasks"]
    },
    "plate": {
        "pick up": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"],
        "wipe": ["household/cleaning"]
    },
    "spoon": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils"],
        "stir": ["household/kitchen_tasks"],
        "scoop": ["household/kitchen_tasks"]
    },
    "fork": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils", "manipulation/lift"],
        "pierce": ["household/kitchen_tasks"],
        "place": ["manipulation/pick_and_place"]
    },
    "knife": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils"],
        "cut": ["household/kitchen_tasks"],
        "place": ["manipulation/pick_and_place"]
    },
    
    # Furniture and large objects
    "table": {
        "wipe": ["household/cleaning"],
        "push": ["manipulation/push"],
        "navigate to": ["navigation/indoor_navigation"]
    },
    "chair": {
        "push": ["manipulation/push", "household/furniture"],
        "pull": ["manipulation/push", "household/furniture"],
        "navigate to": ["navigation/indoor_navigation"]
    },
    "door": {
        "open": ["household/furniture", "utokyo_pr2_opening_fridge"],
        "close": ["household/furniture", "utokyo_pr2_opening_fridge"],
        "push": ["manipulation/push"]
    },
    "drawer": {
        "open": ["household/furniture"],
        "close": ["household/furniture"],
        "pull": ["manipulation/push"]
    },
    
    # Small objects
    "pen": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils", "manipulation/lift"],
        "write with": ["special_objects/utensils"],
        "place": ["manipulation/pick_and_place"]
    },
    "pencil": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils", "aloha_static_pro_pencil"],
        "write with": ["special_objects/utensils"],
        "place": ["manipulation/pick_and_place"]
    },
    "book": {
        "pick up": ["manipulation/pick_and_place"],
        "open": ["manipulation/pick_and_place"],
        "close": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"]
    },
    "phone": {
        "pick up": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"]
    },
    "remote": {
        "pick up": ["manipulation/pick_and_place"],
        "press button": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"]
    },
    
    # Kitchen specific
    "pot": {
        "pick up": ["household/kitchen_tasks"],
        "place": ["household/kitchen_tasks"],
        "fill": ["household/kitchen_tasks", "manipulation/pour"],
        "wash": ["household/kitchen_tasks", "aloha_mobile_wash_pan"]
    },
    "pan": {
        "pick up": ["household/kitchen_tasks"],
        "place": ["household/kitchen_tasks"],
        "tilt": ["household/kitchen_tasks"],
        "wash": ["household/kitchen_tasks", "aloha_mobile_wash_pan"]
    },
    
    # Clothing
    "shirt": {
        "pick up": ["household/clothing"],
        "fold": ["household/clothing", "unitreeh1_fold_clothes"],
        "place": ["household/clothing"]
    },
    "towel": {
        "pick up": ["household/clothing", "aloha_static_towel"],
        "fold": ["household/clothing"],
        "wipe with": ["household/cleaning"]
    },
    
    # Misc
    "box": {
        "pick up": ["manipulation/pick_and_place"],
        "open": ["manipulation/pick_and_place"],
        "close": ["manipulation/pick_and_place"],
        "place": ["manipulation/pick_and_place"]
    },
    "ball": {
        "pick up": ["manipulation/pick_and_place", "special_objects/small_objects"],
        "roll": ["manipulation/push"],
        "throw": ["manipulation/pick_and_place"]
    },
    "bottle cap": {
        "pick up": ["manipulation/pick_and_place", "special_objects/small_objects"],
        "twist": ["household/kitchen_tasks"],
        "place": ["manipulation/pick_and_place"]
    },
    "cable": {
        "pick up": ["special_objects/cables"],
        "insert": ["manipulation/insertion"],
        "route": ["special_objects/cables", "berkeley_cable_routing"]
    },
    "screwdriver": {
        "pick up": ["manipulation/pick_and_place", "special_objects/utensils", "aloha_static_screw_driver"],
        "twist": ["special_objects/utensils"],
        "place": ["manipulation/pick_and_place"]
    },
    "battery": {
        "pick up": ["manipulation/pick_and_place", "aloha_static_battery"],
        "insert": ["manipulation/insertion"],
        "place": ["manipulation/pick_and_place"]
    },
    "tape": {
        "pick up": ["manipulation/pick_and_place", "aloha_static_tape"],
        "cut": ["special_objects/utensils"],
        "place": ["manipulation/pick_and_place"]
    }
}

# Define a fallback mapping for unknown objects or actions
FALLBACK_DATASETS = {
    "default_pick_place": ["lerobot/metaworld_mt50", "lerobot/ucsd_pick_and_place_dataset"],
    "default_push": ["lerobot/metaworld_mt50_push_v2_image", "lerobot/pusht"],
    "default_navigation": ["lerobot/berkeley_gnm_recon"],
    "default_general": ["lerobot/libero_10_image", "lerobot/berkeley_rpt"]
} 