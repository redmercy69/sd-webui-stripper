GENERIC_LIGHTING_NEG_PROMPT = "bad lighting"
CLOTHES_NEG_PROMPT = "clothes, jacket, shirt, bra, bra straps, shoes, chain, necklace, jewelry, piercings"

def generate_generic_body_part_neg_prompt(body_part):
    return f"bad {body_part}, extra {body_part}, imperfect {body_part}, bad {body_part} proportions, flawed {body_part}, covered {body_part}, deformed {body_part}, unrealistic {body_part}, gross {body_part}, malformed {body_part}, mutated {body_part}, fused {body_part}, ugly {body_part}, poorly drawn {body_part}"


GENERIC_ANATOMY_NEG_PROMPT = f"mutated body parts, deformed body features, bad anatomy, Amputee, Mutated, Mutation, Cloned body, Gross proportions, Body horror, Dismembered, Duplicate, Improper scale, Missing limbs, extra limbs, malformed limbs, disfigured, ugly body"

POS_PROMPT_FORMAT = "RAW photo, nude [pregnant] [body type] [name], age [age], [neck], [shoulders], [arms], [hands], [back], [collar bone], [chest], [boobs], [nipples], [waist], [belly], [vagina], [ass], [hips], [thighs], [knees], [shins], [feet], soft lighting, smooth symmetrical body"
NEG_PROMPT_FORMAT = f"{CLOTHES_NEG_PROMPT}, [pregnant], [body type], [neck], [shoulders], [arms], [hands], [back], [collar bone], [chest], [boobs], [nipples], [belly], [vagina], [ass], [thighs], [knees], [shins], [feet], {GENERIC_ANATOMY_NEG_PROMPT}, {GENERIC_LIGHTING_NEG_PROMPT}, text"

HEIGHTS     = ["Auto", "4-foot", "5-foot", "6-foot"]
BODY_TYPES  = ["Auto", "anorexic", "petite", "slim", "muscular", "fat", "obese"]
BOOB_SIZES  = ["Auto", "flat", "tiny", "small", "medium", "big", "giant"]
ABS_OPTIONS = ["Auto", "no abs", "2-pack-abs", "4-pack-abs", "6-pack-abs"]
ASS_SIZES   = ["Auto", "flat", "tiny", "small", "medium", "big", "giant"]

def generate_stripper_prompts(age:int, body_type:str, neck_is_visible:bool, 
                                shoulders_are_visible:bool, arms_are_visible:bool, hands_are_visible:bool, 
                                collar_bone_is_visible:bool, chest_is_visible:bool, back_is_visible:bool, 
                                boobs_are_visible:bool, boobs_size:str, 
                                nipples_are_visible:bool, hard_nipples:bool, 
                                waist_is_visible:bool, narrow_waist:bool, 
                                belly_is_visible:bool, belly_button_is_visible:bool, abs_option:str, is_pregnant:bool,
                                vagina_is_visible:bool, pubic_hair:bool,
                                ass_is_visible:bool, ass_size:str,
                                hips_are_visible:bool,
                                thighs_are_visible:bool, knees_are_visible:bool, shins_are_visible:bool, feet_are_visible:bool):

    pos_prompt = generate_stripper_pos_prompt(age, body_type, neck_is_visible, 
                                shoulders_are_visible, arms_are_visible, hands_are_visible, 
                                collar_bone_is_visible, chest_is_visible, back_is_visible, 
                                boobs_are_visible, boobs_size, 
                                nipples_are_visible, hard_nipples, 
                                waist_is_visible, narrow_waist, 
                                belly_is_visible, belly_button_is_visible, abs_option, is_pregnant,
                                vagina_is_visible, pubic_hair,
                                ass_is_visible, ass_size,
                                hips_are_visible,
                                thighs_are_visible, knees_are_visible, shins_are_visible, feet_are_visible)

    neg_prompt = generate_stripper_neg_prompt(age, body_type, neck_is_visible, 
                                shoulders_are_visible, arms_are_visible, hands_are_visible, 
                                collar_bone_is_visible, chest_is_visible, back_is_visible, 
                                boobs_are_visible, boobs_size, 
                                nipples_are_visible, hard_nipples, 
                                waist_is_visible, narrow_waist, 
                                belly_is_visible, belly_button_is_visible, abs_option, is_pregnant,
                                vagina_is_visible, pubic_hair,
                                ass_is_visible, ass_size,
                                hips_are_visible,
                                thighs_are_visible, knees_are_visible, shins_are_visible, feet_are_visible)
    return pos_prompt, neg_prompt

def generate_stripper_pos_prompt(age:int, body_type:str, neck_is_visible:bool, 
                                shoulders_are_visible:bool, arms_are_visible:bool, hands_are_visible:bool, 
                                collar_bone_is_visible:bool, chest_is_visible:bool, back_is_visible:bool, 
                                boobs_are_visible:bool, boobs_size:str, 
                                nipples_are_visible:bool, hard_nipples:bool, 
                                waist_is_visible:bool, narrow_waist:bool, 
                                belly_is_visible:bool, belly_button_is_visible:bool, abs_option:str, is_pregnant:bool,
                                vagina_is_visible:bool, pubic_hair:bool,
                                ass_is_visible:bool, ass_size:str,
                                hips_are_visible:bool,
                                thighs_are_visible:bool, knees_are_visible:bool, shins_are_visible:bool, feet_are_visible:bool):
    pos_prompt = POS_PROMPT_FORMAT

    #Name
    if age >= 20:
        name = "woman"
    else:
        name = "girl"

    pos_prompt = pos_prompt.replace("[pregnant]", "pregnant") if is_pregnant  else pos_prompt.replace(" [pregnant]", "")

    pos_prompt = pos_prompt.replace("[name]", name)
  
    pos_prompt = pos_prompt.replace("[age]", str(age))

    #Body Type
    if body_type != "Auto":
        pos_prompt = pos_prompt.replace("[body type]", body_type)
    else:
        pos_prompt = pos_prompt.replace(" [body type]", "")

    pos_prompt = pos_prompt.replace("[neck]", "neck") if neck_is_visible                      else pos_prompt.replace(", [neck]", "")
    pos_prompt = pos_prompt.replace("[back]", "back") if back_is_visible                      else pos_prompt.replace(", [back]", "")
    pos_prompt = pos_prompt.replace("[shoulders]", "shoulders") if shoulders_are_visible      else pos_prompt.replace(", [shoulders]", "")
    pos_prompt = pos_prompt.replace("[arms]", "arms") if arms_are_visible                     else pos_prompt.replace(", [arms]", "")
    pos_prompt = pos_prompt.replace("[hands]", "hands") if hands_are_visible                  else pos_prompt.replace(", [hands]", "")
    pos_prompt = pos_prompt.replace("[collar bone]", "collar bone") if collar_bone_is_visible else pos_prompt.replace(", [collar bone]", "")
    pos_prompt = pos_prompt.replace("[chest]", "chest") if chest_is_visible                   else pos_prompt.replace(", [chest]", "")
    pos_prompt = pos_prompt.replace("[hips]", "hips") if hips_are_visible                     else pos_prompt.replace(", [hips]", "")

    #Boobs
    if boobs_are_visible and boobs_size != "flat":
        if boobs_size == "Auto":
            pos_prompt = pos_prompt.replace("[boobs]", "boobs")
        else:
            pos_prompt = pos_prompt.replace("[boobs]", f"{boobs_size} boobs")
    else:
        pos_prompt = pos_prompt.replace(", [boobs]", "")

    #Niples
    if nipples_are_visible:
        if hard_nipples:
            pos_prompt = pos_prompt.replace("[nipples]", "hard nipples")
        else:
            pos_prompt = pos_prompt.replace("[nipples]", "soft nipples")

    else:
        pos_prompt = pos_prompt.replace(", [nipples]", "")

    #Waist
    if waist_is_visible:
        waist_prompt = "waist"

        if narrow_waist:
            waist_prompt = "narrow waist"

        pos_prompt = pos_prompt.replace("[waist]", waist_prompt)
    else:
        pos_prompt = pos_prompt.replace(", [waist]", "")

    #Belly
    if belly_is_visible:
        belly_prompt = "belly"

        #Belly Button
        if belly_button_is_visible:
            belly_prompt += ", belly button"

        #Abs
        if abs_option != "Auto":
            if abs_option != "no abs":
                belly_prompt += f", {abs_option}"

        pos_prompt = pos_prompt.replace("[belly]", belly_prompt)
    else:
        pos_prompt = pos_prompt.replace(", [belly]", "")

    #Vagina
    if vagina_is_visible:
        vagina_prompt:str = "vagina"

        if pubic_hair:
            vagina_prompt += ", pubic hair"

        pos_prompt = pos_prompt.replace("[vagina]", vagina_prompt)
    else:
        pos_prompt = pos_prompt.replace(", [vagina]", "")

    #Ass
    if ass_is_visible:
        if ass_size == "Auto":
            pos_prompt = pos_prompt.replace("[ass]", "ass")
        else:
            pos_prompt = pos_prompt.replace("[ass]", f"{ass_size} ass")
    else:
        pos_prompt = pos_prompt.replace(", [ass]", "")
    #Legs
        #Thighs
    if thighs_are_visible:
        pos_prompt = pos_prompt.replace("[thighs]", "thighs")
    else:
        pos_prompt = pos_prompt.replace(", [thighs]", "")
    
        #Knees
    if knees_are_visible:
        pos_prompt = pos_prompt.replace("[knees]", "knees")
    else:
        pos_prompt = pos_prompt.replace(", [knees]", "")

        #Shins
    if shins_are_visible:
        pos_prompt = pos_prompt.replace("[shins]", "shin")
    else:
        pos_prompt = pos_prompt.replace(", [shins]", "")

        #Feet
    if feet_are_visible:
        pos_prompt = pos_prompt.replace("[feet]", "feet")
    else:
        pos_prompt = pos_prompt.replace(", [feet]", "")
    
    return pos_prompt

def generate_stripper_neg_prompt(age:int, body_type:str, neck_is_visible:bool, 
                                shoulders_are_visible:bool, arms_are_visible:bool, hands_are_visible:bool, 
                                collar_bone_is_visible:bool, chest_is_visible:bool, back_is_visible:bool, 
                                boobs_are_visible:bool, boobs_size:str, 
                                nipples_are_visible:bool, hard_nipples:bool, 
                                waist_is_visible:bool, narrow_waist:bool, 
                                belly_is_visible:bool, belly_button_is_visible:bool, abs_option:str, is_pregnant:bool,
                                vagina_is_visible:bool, pubic_hair:bool,
                                ass_is_visible:bool, ass_size:str,
                                hips_are_visible:bool,
                                thighs_are_visible:bool, knees_are_visible:bool, shins_are_visible:bool, feet_are_visible:bool):
    neg_prompt = NEG_PROMPT_FORMAT
    
    neg_prompt = neg_prompt.replace("[pregnant]", "pregnant") if not is_pregnant else neg_prompt.replace(" [pregnant]", "")

    #Body Type
    if body_type != "Auto":
        body_type_n_prompt = f"{generate_generic_body_part_neg_prompt('body')}, "
        for x in BODY_TYPES:
            if x != "Auto" and x != body_type:
                if BODY_TYPES.index(x) != len(BODY_TYPES) - 1:
                    body_type_n_prompt += f"{x}, "
                else:
                    body_type_n_prompt += x

        neg_prompt = neg_prompt.replace("[body type]", body_type_n_prompt)
    else:
        neg_prompt = neg_prompt.replace(", [body type]", "")

    #Neck
    if neck_is_visible:
        neck_neg_prompt = generate_generic_body_part_neg_prompt("neck")

        if hard_nipples:
            neck_neg_prompt += ", soft nipples"
        else:
            neck_neg_prompt += ", hard nipples"

        neg_prompt = neg_prompt.replace("[neck]", neck_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[neck]", "neck")

    back_neg_prompt = generate_generic_body_part_neg_prompt("back")
    neg_prompt = neg_prompt.replace("[back]", back_neg_prompt) if back_is_visible else neg_prompt.replace("[back]", "back")

    #Shoulders
    if shoulders_are_visible:
        shoulders_neg_prompt = generate_generic_body_part_neg_prompt("shoulders")
        neg_prompt = neg_prompt.replace("[shoulders]", shoulders_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[shoulders]", "shoulders")

    #Arms
    if arms_are_visible:
        arms_neg_prompt = generate_generic_body_part_neg_prompt("arms")
        neg_prompt = neg_prompt.replace("[arms]", arms_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[arms]", "arms")

    #Hands
    if hands_are_visible:
        hands_neg_prompt = generate_generic_body_part_neg_prompt("hands")
        neg_prompt = neg_prompt.replace("[hands]", hands_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[hands]", "hands")

    #Collar Bone
    if collar_bone_is_visible:
        collar_bone_neg_prompt = generate_generic_body_part_neg_prompt("collar bone")
        neg_prompt = neg_prompt.replace("[collar bone]", collar_bone_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[collar bone]", "collar bone")

    #Chest
    if chest_is_visible:
        chest_neg_prompt = generate_generic_body_part_neg_prompt("chest")
        neg_prompt = neg_prompt.replace("[chest]", chest_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[chest]", "chest")

    #Boobs
    if boobs_are_visible:
        boobs_neg_prompt = generate_generic_body_part_neg_prompt("boobs")
        
        for size in BOOB_SIZES:
            if size != boobs_size and size != "Auto":
                if size == "flat":
                    boobs_neg_prompt += ", flat chest"
                else:
                    boobs_neg_prompt += f", {size}"

        neg_prompt = neg_prompt.replace("[boobs]", boobs_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[boobs]", "boobs")

    #Nipples
    if nipples_are_visible:
        nipples_neg_prompt = generate_generic_body_part_neg_prompt("nipples") + ", more than 1 nipple per boob"
        neg_prompt = neg_prompt.replace("[nipples]", nipples_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[nipples]", "nipples")

    #Waist
    if waist_is_visible:
        waist_neg_prompt = ""

        if not narrow_waist:
            waist_neg_prompt += "narrow waist"
        else:
            waist_neg_prompt += "waist"

        waist_neg_prompt += f", {generate_generic_body_part_neg_prompt('waist')}"

        neg_prompt = neg_prompt.replace("[waist]", waist_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[waist]", "waist")

    #Belly
    if belly_is_visible:
        belly_neg_prompt = generate_generic_body_part_neg_prompt("belly")
        
        #Abs
        for x in ABS_OPTIONS:
            if x != "Auto":
                belly_neg_prompt += f", {x}"

        belly_neg_prompt += f", {generate_generic_body_part_neg_prompt('belly button')}" if belly_button_is_visible else ", belly button"

        neg_prompt = neg_prompt.replace("[belly]", belly_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[belly]", "belly")

    #Vagina
    if vagina_is_visible:
        vagina_neg_prompt = "multiple vaginas per girl, " + generate_generic_body_part_neg_prompt("vagina")

        if pubic_hair:
            vagina_neg_prompt += f", {generate_generic_body_part_neg_prompt('pubic hair')}"
        else:
            vagina_neg_prompt += ", pubic hair"

        neg_prompt = neg_prompt.replace("[vagina]", vagina_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[vagina]", "vagina")

    #Ass
    if ass_is_visible:
        ass_neg_prompt = generate_generic_body_part_neg_prompt("ass")

        if ass_size != "Auto":
            for x in ASS_SIZES:
                if x != ass_size:
                    ass_neg_prompt += f", {x}"
                
        neg_prompt = neg_prompt.replace("[ass]", ass_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[ass]", "ass")

    #Hips
    if hips_are_visible:
        hips_prompt = generate_generic_body_part_neg_prompt("hips")
        neg_prompt = neg_prompt.replace("[hips]", hips_prompt)
    else:
        neg_prompt = neg_prompt.replace("[hips]", "ass")

    #Legs
        #Thighs
    if thighs_are_visible:
        thighs_neg_prompt = generate_generic_body_part_neg_prompt("thighs")
        neg_prompt = neg_prompt.replace("[thighs]", thighs_neg_prompt)
    else:
        neg_prompt = neg_prompt.replace("[thighs]", "thighs")

        #Knees
    if knees_are_visible:
        knees_prompt = generate_generic_body_part_neg_prompt("knees")
        neg_prompt = neg_prompt.replace("[knees]", knees_prompt)
    else:
        neg_prompt = neg_prompt.replace("[knees]", "knees")

        #Shins
    if shins_are_visible:
        shins_prompt = generate_generic_body_part_neg_prompt("shins")
        neg_prompt = neg_prompt.replace("[shins]", shins_prompt)
    else:
        neg_prompt = neg_prompt.replace("[shins]", "shins")

        #Feet
    if feet_are_visible:
        feet_prompt = generate_generic_body_part_neg_prompt("feet") + generate_generic_body_part_neg_prompt("toes")
        neg_prompt = neg_prompt.replace("[feet]", feet_prompt)
    else:
        neg_prompt = neg_prompt.replace("[feet]", "feet")

    return neg_prompt
