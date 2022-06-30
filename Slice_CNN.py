from argparse import ArgumentParser
from os.path import exists

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, cat
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ADNI_Image_Module import ADNI_Image_Module
from ADNI_Model import ADNI_Model


class ADNI_3slice_CNN(ADNI_Model):
    def __init__(self):
        super().__init__()
        self.num_classes = 2

        self.conv_part = nn.Sequential(
            self._conv_layer_set(1, 16, 5, 1),
            self._conv_layer_set(16, 16, 5, 2),
            self._conv_layer_set(16, 16, 3, 1),
            self._conv_layer_set(16, 16, 3, 2),
            self._conv_layer_set(16, 16, 3, 1),
            self._conv_layer_set(16, 16, 3, 2),
            self._conv_layer_set(16, 16, 3, 1),
            self._conv_layer_set(16, 16, 3, 2),
            nn.Flatten(),
        )

        self.feature_extractor_fully_connected = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3072, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(32, self.num_classes if self.num_classes > 2 else 1)

    @staticmethod
    def _conv_layer_set(in_c, out_c, ks, strides):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(ks, ks), padding=0, stride=strides),
            nn.ReLU(),
        )
        return conv_layer

    def forward(self, x):
        slice1_features = self.conv_part(x[:, 0, :, :, :])
        slice2_features = self.conv_part(x[:, 1, :, :, :])
        slice3_features = self.conv_part(x[:, 2, :, :, :])
        final_features = self.feature_extractor_fully_connected(
            cat([slice1_features, slice2_features, slice3_features], dim=1))
        prediction = self.classifier(final_features)
        return prediction

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-4)

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'loss/val'
        }
        return [optimizer], [scheduler]

    def __str__(self):
        return 'ADNI_3slice_CNN'


if __name__ == '__main__':
    parser = ArgumentParser()
    # nargs="+" would enable multi-gpu, but results seemed to differ when using different amounts of gpus
    parser.add_argument("-g", "--gpu", dest="gpu", default=2, help="GPU to use", type=int)
    parser.add_argument("-r", "--ratio", dest="ratio", default=0.5,
                        help="Ratio of females in training dataset", type=float)
    parser.add_argument("-i", "--run_idces", dest="run_idces", default=[0, 1, 2, 3, 4], nargs="+",
                        help="Run idces to iterate over", type=int)
    parser.add_argument("-e", "--export_path", dest="export_path", default=None, type=str)
    parser.add_argument("-a", "--feature_csv_dir", dest="feature_csv_dir", type=str, default="")
    parser.add_argument("-d", "--split_dir", dest="split_dir", type=str, default="/dtu-compute/ADNIbias/ewipe/splits/")
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default="/dtu-compute/ADNIbias/ewipe/CNN-logs/")
    parser.add_argument("-c", "--chkpt_dir", dest="chkpt_dir", type=str,
                        default="/dtu-compute/ADNIbias/ewipe/CNN-chkpts/")
    args = parser.parse_args()

    assert (isinstance(args.ratio, float))

    # Yes, the ADNI3 images are in the ADNI1 directory for some reason
    image_paths = ["/scratch/ewipe/freesurfer_ADNI1",
                   "/scratch/ewipe/freesurfer_ADNI2",
                   "/scratch/ewipe/freesurfer_ADNI1"]

    for run_idx in args.run_idces:

        chkpt_file = args.chkpt_dir + f'ADNI_3slice_CNN-ratio={args.ratio:.2f}-run={run_idx}.ckpt'

        if not exists(chkpt_file):
            adhc_split_csvs = [
                args.split_dir + f'adhc12_Sex_{run_idx}_{args.ratio:.2f}_0_train.csv',
                args.split_dir + f'adhc12_Sex_{run_idx}_{args.ratio:.2f}_0_val.csv',
                args.split_dir + f'adhc12_Sex_{run_idx}_test.csv',
            ]
            adni1_dm = ADNI_Image_Module(image_paths=image_paths, adni_set=3, batch_size=6,
                                         adhc_split_csvs=adhc_split_csvs, increased_aug=True,
                                         export_path=args.export_path, sliced=True,
                                         feature_csv_dir=args.feature_csv_dir)

            mdl = ADNI_3slice_CNN()
            log_name = f'3slice_CNN-r{args.ratio:.2f}'

            tb_logger = TensorBoardLogger(args.log_dir, name=log_name, version=f'test set {run_idx}')

            trainer = Trainer(
                logger=tb_logger,
                max_epochs=200,
                gpus=[args.gpu],
                precision=16,
                callbacks=[StochasticWeightAveraging(), EarlyStopping(monitor="loss/val", patience=60)],
                gradient_clip_val=1.0,
                enable_checkpointing=False,
                log_every_n_steps=26)

            trainer.fit(mdl, adni1_dm)
            trainer.save_checkpoint(chkpt_file)
