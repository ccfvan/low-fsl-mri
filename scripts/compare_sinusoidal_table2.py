"""Rerun the three continuous-waveform sinusoidal Table 2 rows, without plots."""
import sys
from pathlib import Path
import argparse
import csv
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lowfsl.sinusoidal_continuous import continuous_signals
from lowfsl.engine import fit_exp_batch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt-us', type=float, default=25)
    parser.add_argument('--output', default='sinusoidal_table2.csv')
    args = parser.parse_args()
    b0, b1 = np.meshgrid(np.arange(-250,251,5), np.arange(-25,25.01,.5))
    references = {20: (38.31,80.15), 50: (45.80,95.56), 100: (47.35,111.57)}
    rows = []
    for f in (20,50,100):
        signals = continuous_signals(f,b0.ravel(),1+b1.ravel()/100,args.dt_us*1e-6)
        fit = fit_exp_batch(np.array([0,2,4,8])/f, abs(signals))
        error = abs(100*(fit['Tfit_s']/.08-1))
        med,p95 = np.median(error), np.percentile(error,95)
        row = dict(target_Hz=f,median_abs_error_pct=float(med),p95_abs_error_pct=float(p95),
                   reference_median_pct=references[f][0],reference_p95_pct=references[f][1],
                   matches_table_rounding=bool(round(float(med),2)==references[f][0] and round(float(p95),2)==references[f][1]),
                   max_dt_us=args.dt_us,b0_window_Hz=250,b1_window_pct=25)
        print(row,flush=True);rows.append(row)
    with open(args.output,'w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
