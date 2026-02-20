import torch
import pandas as pd
from stavki.models.deep_interaction_wrapper import DeepInteractionWrapper

def main():
    w = DeepInteractionWrapper()
    w.load_checkpoint("models/deep_interaction_v3.pth")

    row = pd.Series({
        'HomeTeam': 'Arsenal', 'AwayTeam': 'Chelsea', 'Date': '2024-04-23',
        'league_id': 8, 'home_team_id': 19, 'away_team_id': 18
    })

    # Hook the forward method to log shapes of the 14 tensors before torch.cat
    old_forward = w.network.forward
    def new_forward(h_players, a_players, h_positions, a_positions, 
                    h_manager, a_manager,
                    h_ctx, a_ctx, h_mom, a_mom,
                    league_id, venue_id, referee_id, season_phase, h2h_features):
        
        # We manually execute the forward pass and print shapes
        net = w.network
        h_vec = net.player_encoder(h_players, h_positions)
        a_vec = net.player_encoder(a_players, a_positions)
        matchup = net.cross_interaction(
            h_players, a_players, h_positions, a_positions,
            net.player_encoder.embedding, net.player_encoder.position_enc
        )
        league_vec = net.league_encoder(league_id)
        venue_vec = net.venue_encoder(venue_id)
        ref_vec = net.referee_encoder(referee_id)
        season_vec = net.season_encoder(season_phase)
        h2h_vec = net.h2h_encoder(h2h_features)
        h_ctx_vec = net.context_encoder(h_ctx)
        a_ctx_vec = net.context_encoder(a_ctx)
        h_mom_decayed = net.time_decay(h_mom)
        a_mom_decayed = net.time_decay(a_mom)
        h_mom_vec = net.momentum_encoder(h_mom_decayed)
        a_mom_vec = net.momentum_encoder(a_mom_decayed)
        h_man_vec = net.manager_encoder(h_manager)
        a_man_vec = net.manager_encoder(a_manager)
        
        print(f"h_vec: {h_vec.shape}")
        print(f"a_vec: {a_vec.shape}")
        print(f"matchup: {matchup.shape}")
        print(f"h_ctx_vec: {h_ctx_vec.shape}")
        print(f"a_ctx_vec: {a_ctx_vec.shape}")
        print(f"h_mom_vec: {h_mom_vec.shape}")
        print(f"a_mom_vec: {a_mom_vec.shape}")
        print(f"league_vec: {league_vec.shape}")
        print(f"venue_vec: {venue_vec.shape}")
        print(f"ref_vec: {ref_vec.shape}")
        print(f"season_vec: {season_vec.shape}")
        print(f"h2h_vec: {h2h_vec.shape}")
        print(f"h_man_vec: {h_man_vec.shape}")
        print(f"a_man_vec: {a_man_vec.shape}")
        
        # Test original
        try:
            return old_forward(h_players, a_players, h_positions, a_positions, 
                               h_manager, a_manager,
                               h_ctx, a_ctx, h_mom, a_mom,
                               league_id, venue_id, referee_id, season_phase, h2h_features)
        except Exception as e:
            print("ERROR IN FORWARD:", e)
            
    w.network.forward = new_forward
    w._predict_single(row, 0)

if __name__ == "__main__":
    main()
