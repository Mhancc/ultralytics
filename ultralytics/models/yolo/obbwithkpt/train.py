# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBWithHtpModel, OBBWithKptModel
from ultralytics.utils import DEFAULT_CFG, RANK, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results

class OBBWithKptTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obbwithkpt"
        super().__init__(cfg, overrides, _callbacks)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = OBBWithKptModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"],verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "kpt_loss", "kobj_loss", "dfl_loss"
        return yolo.obbwithkpt.OBBWithKptValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
    
    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # Freeze specified layers
        freeze_header_names = ["model.42.cv3","model.42.cv5"]
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_header_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True


class OBBWithHtpTrainer(OBBWithKptTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml', epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obbwithhtp"
        super(OBBWithKptTrainer,self).__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = OBBWithHtpModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"],verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "kpt_loss", "dfl_loss"
        return yolo.obbwithkpt.OBBWithHtpValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    


