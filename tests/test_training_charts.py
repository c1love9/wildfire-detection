import unittest
from training_charts import history_svg


class TrainingChartTests(unittest.TestCase):
    def test_every_real_epoch_retained_and_changed_job_changes_plot(self):
        rows=[{'epoch':i,'train/box_loss':i/10} for i in range(1,51)]
        svg=history_svg(rows,['train/box_loss'],'训练损失')
        self.assertEqual(svg.count('<circle '),50)
        self.assertIn('Epoch 50',svg)
        self.assertNotIn('data-',svg)
        changed=[{**r,'train/box_loss':(51-r['epoch'])/10} for r in rows]
        self.assertNotEqual(svg,history_svg(changed,['train/box_loss'],'训练损失'))

    def test_unknown_value_breaks_line_instead_of_imputing_zero(self):
        rows=[{'epoch':i,'loss':v} for i,v in [(1,2),(2,None),(3,1),(4,float('nan'))]]
        svg=history_svg(rows,['loss'],'<损失>')
        self.assertEqual(svg.count('<circle '),2)
        self.assertIn('&lt;损失&gt;',svg)
        self.assertIn('d="M64.00',svg)
        self.assertNotIn('nan',svg)

    def test_single_epoch_and_empty_history(self):
        self.assertEqual(history_svg([],['loss'],'Loss'),'')
        self.assertIn('Epoch 1',history_svg([{'epoch':1,'loss':0}],['loss'],'Loss'))


if __name__=='__main__':unittest.main()
