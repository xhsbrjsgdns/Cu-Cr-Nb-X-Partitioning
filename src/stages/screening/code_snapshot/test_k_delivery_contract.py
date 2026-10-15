from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
class DeliveryContract(unittest.TestCase):
    def test_nb_confirmation(self):
        d = pd.read_csv(ROOT/'05_screening_confirmation/STEP5K_SCREENING_CALPHAD_CONFIRMATION.csv')
        self.assertIn('pred_Nb_P_FCC', d)
        self.assertIn('calphad_Nb_P_C15', d)
        self.assertEqual(len(d), 20)
    def test_literature_schema(self):
        d = pd.read_csv(ROOT/'02_external_validation/STEP5K_EXTERNAL_LITERATURE_VALIDATION.csv')
        self.assertTrue({'comparison_class','qualitative_agreement','source_quote_or_location','composition_basis','CALPHAD_raw_phases'}.issubset(d.columns))
        self.assertTrue(set(d.qualitative_agreement) <= {'CONSISTENT','PARTIALLY_CONSISTENT','DISCREPANT_BUT_EXPLAINABLE','NOT_COMPARABLE'})
    def test_requested_figures_and_insertions(self):
        names=['virtual_screening_space','global_pareto_front','family_specific_pareto','Cr_retention_vs_release','C15_tradeoff','finalist_composition_map','ML_vs_CALPHAD_confirmation','candidate_topology_heatmap']
        for i,n in enumerate(names,1):
            self.assertTrue((ROOT/f'06_figures/Fig_K_{i:02d}_{n}.png').exists(), n)
        for name in ['RESULTS','DISCUSSION','METHODS','SUPPLEMENTARY']:
            self.assertTrue((ROOT/f'08_manuscript_insertions/STEP5K_JMST_{name}_INSERTIONS.md').exists())
if __name__=='__main__': unittest.main()
