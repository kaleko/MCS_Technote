import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.stats import norm
from root_numpy import root2array
from scipy.optimize import curve_fit
from lmfit import  Model #better gaussian model


titles = { 
           'SingleMuonMCTrack'        : 'Fully Contained Single Muon MCTracks',
           'SingleMuonRecoTrack'      : 'Fully Contained, Well Reconstructed Single Muon Tracks',
           'DataBNBSelectedRecoTrack' : 'Selected, Well Reconstructed Tracks from NumuCC Data',
           'MCBNBSelectedRecoTrack'   : 'Selected, Well Reconstructed Tracks from NumuCC Simulation',
           'MCBNBSelectedRecoTrackNominalHL'   : 'Selected, Well Reconstructed Tracks from NumuCC Simulation (Nominal Highland)',
           'MCBNBRecoTrack'           : 'MC numuCC BNB Truth-Selected, Well Reconstructed Tracks',
           'MCBNBMCTrack'             : 'MC numuCC BNB Truth-Selected MCTracks',
           'MCBNBMCTrackExiting'      : 'MC numuCC BNB Truth-Selected EXITING MCTracks',
           'MCBNBRecoTrackExiting'    : 'MC numuCC BNB Truth-Selected EXITING Reco Tracks',
           'full_MCS_energy'          : 'MCS Total Energy [GeV]',
           'full_range_energy'        : 'Range-Based Total Energy [GeV]',
           'full_integrated_range_energy'        : 'Integrated Range-Based Total Energy [GeV]',
           'full_MCS_momentum'        : 'MCS Momentum [GeV/c]',
           'full_range_momentum'      : 'Range-Based Momentum [GeV/c]',
           'full_integrated_range_momentum'      : 'Integrated Range-Based Momentum [GeV/c]',
           'full_MCS_momentum_inverse'        : 'Inverse MCS Momentum [GeV^-1]',
           'full_range_momentum_inverse'      : 'Inverse Range-Based Momentum [GeV^-1]',
           'true_E'                   : 'True Total Energy [GeV]',
           'true_momentum'                   : 'True Momentum [GeV/c]',
           'true_momentum_inverse'                   : 'Inverse True Momentum [GeV^-1]',
           'full_length'              : 'Length Of Track Contained [cm]'
         }

latextitles = {
           'full_range_energy'   : 'E_{Range}',
           'full_range_momentum' : 'p_{Range}',
           'full_integrated_range_energy'   : 'E_{Integrated Range}',
           'full_integrated_range_momentum' : 'p_{Integrated Range}',
           'full_MCS_energy'     : 'E_{MCS}',
           'full_MCS_momentum'   : 'p_{MCS}',
           'full_MCS_momentum_inverse'        : 'p_{MCS}^{-1}',
           'full_range_momentum_inverse'      : 'p_{Range}^{-1}',
           'true_E'              : 'E_{True}',
           'true_momentum'              : 'p_{True}',
           'true_momentum_inverse'              : 'p_{True}^{-1}',
           'full_length'         : 'L_{Track}'
         }

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))

def get_dfs(myfile):
    #This df has track-by-track information (MCS energy, range energy, etc)
    df = pd.DataFrame( root2array ( myfile, 'MCS_bias_tree' ) )
    df['full_MCS_momentum_inverse'] = 1./df['full_MCS_momentum']
    df['full_range_momentum_inverse'] = 1./df['full_range_momentum']
    
    if 'true_momentum' in df.columns.values:
        df['true_momentum_inverse'] = 1./df['true_momentum']

    #This df has segment-by-segment deviation (scattering angle, etc)
    segdf = pd.DataFrame(  root2array ( myfile, 'TMC_debug_tree' ) )
    segdf['dthetayoverpredictedRMS'] = segdf['delta_theta_y']/segdf['predicted_RMS']
    segdf['dthetayovertruepredictedRMS'] = segdf['delta_theta_y']/segdf['true_predicted_RMS']
    #segdf['dthetayovertruepredictedRMS_momentumdepenentconstant'] = segdf['delta_theta_y']/segdf['true_predicted_RMS_momentumdependentconstant']
    segdf['dthetayoverpredictedRMS_fromMCS'] = segdf['delta_theta_y']/segdf['predicted_RMS_fromMCS']
    segdf['tempdenom'] = np.sqrt(np.square(segdf['predicted_RMS_fromMCS']) + np.square(3.))
    segdf['dthetayoverpredictedRMS_fromMCS_with3res'] = segdf['delta_theta_y']/segdf['tempdenom']
    
    segdf['dthetaxoverpredictedRMS'] = segdf['delta_theta_x']/segdf['predicted_RMS']
    segdf['dthetaxovertruepredictedRMS'] = segdf['delta_theta_x']/segdf['true_predicted_RMS']
    #segdf['dthetaxovertruepredictedRMS_momentumdepenentconstant'] = segdf['delta_theta_x']/segdf['true_predicted_RMS_momentumdependentconstant']
    segdf['dthetaxoverpredictedRMS_fromMCS'] = segdf['delta_theta_x']/segdf['predicted_RMS_fromMCS']
    segdf['dthetaxoverpredictedRMS_fromMCS_with3res'] = segdf['delta_theta_x']/segdf['tempdenom']
    
    
    #Optional driver DF tree that holds some MCTrack informationOA
    driverdf = pd.DataFrame( root2array ( myfile, 'driver_tree' ) )
    
    #Merge it into the main df by run,subrun,eventid
    df = df.merge(driverdf, on=['run','subrun','eventid'])
    #also merge into segdf
    segdf = segdf.merge(driverdf, on=['run','subrun','eventid'])

    return df, segdf
