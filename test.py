import gitcg
import random
import time
# Set players initial deck
DECK0 = gitcg.Deck(characters=[1411, 1510, 2103], cards=[214111, 311503, 214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503,214111, 311503])
DECK1 = gitcg.Deck(characters=[1609, 2203, 1608], cards=[312025, 321002, 312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002,312025, 321002])

class MyPlayer(gitcg.Player):
    # implements on_notify, on_action, etc.
    # See `gitcg.Player`'s documentation for detail.
    pass

class RandomActionPlayer(gitcg.Player):
    who: int
    omni_dice_count = 0 # record dice count of OMNI
    def __init__(self, who: int):
        self.who = who


    def on_choose_active(self, request: gitcg.ChooseActiveRequest) -> gitcg.ChooseActiveResponse:
        return gitcg.ChooseActiveResponse(active_character_id=request.candidate_ids[0])

    def on_notify(self, notification):
        self.omni_dice_count = len([i for i in notification.state.player[self.who].dice if i == gitcg.DiceType.DICE_OMNI])

    def on_action(self, request: gitcg.ActionRequest) -> gitcg.ActionResponse:
        chosen_index = 0
        used_dice: list[gitcg.DiceType] = []
        actions = list(enumerate(request.action))
        random.shuffle(actions)
        # select the first action (from shuffled list) that can be performed by current OMNI dice.
        for i, action in actions:
            required_count = 0
            has_non_dice_requirement = False
            if action.HasField("elemental_tuning"):
                continue
            for req in list(action.required_cost):
                if req.type == gitcg.DiceRequirementType.DICE_REQ_ENERGY or req.type == gitcg.DiceRequirementType.DICE_REQ_LEGEND:
                    has_non_dice_requirement = True
                else:
                    required_count += req.count
            if has_non_dice_requirement:
                continue
            if required_count > self.omni_dice_count:
                continue
            chosen_index = i
            used_dice = [gitcg.DiceType.DICE_OMNI] * required_count
            break
        return gitcg.ActionResponse(chosen_action_index=chosen_index, used_dice=used_dice)
    def on_reroll_dice(self, request: gitcg.RerollDiceRequest) -> gitcg.RerollDiceResponse:
        return gitcg.RerollDiceResponse() # or RerollDiceResponse(dice_to_reroll=[])

    def on_select_card(self, request: gitcg.SelectCardRequest) -> gitcg.SelectCardResponse:
        return gitcg.SelectCardResponse(selected_definition_id=request.candidate_definition_ids[0])

    def on_switch_hands(self, request: gitcg.SwitchHandsRequest) -> gitcg.SwitchHandsResponse:
        return gitcg.SwitchHandsResponse() # or SwitchHandsResponse(removed_hand_ids=[])

# Initialize the game
game = gitcg.Game(create_param=gitcg.CreateParam(deck0=DECK0, deck1=DECK1))
game.set_player(0, RandomActionPlayer(0))
game.set_player(1, RandomActionPlayer(1))

# Start and step the game until end
game.start()
t0 = time.time()
while game.is_running():
    game.step()
    print(game.state().current_turn())
    input()
t1 = time.time()
print(t1-t0)