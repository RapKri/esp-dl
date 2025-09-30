from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Attention
from ultralytics.engine.exporter import Exporter, try_export, arange_patch
from ultralytics.utils import LOGGER, __version__, colorstr
from ultralytics.utils.checks import check_requirements
# from ultralytics.utils.torch_utils import get_latest_opset
import torch
import onnx


def get_latest_opset():
    """Get the latest supported ONNX opset version."""
    # For ONNX 1.17.0 and PyTorch 2.8.0+, opset 18 is well supported
    # You can adjust this based on your target deployment requirements
    return 17  # Conservative choice that works well with most systems


class ESP_Detect(Detect):
    def forward(self, x):
        """Returns predicted bounding boxes and class probabilities respectively."""
        # self.nl = 3
        box0 = self.cv2[0](x[0])
        score0 = self.cv3[0](x[0])

        box1 = self.cv2[1](x[1])
        score1 = self.cv3[1](x[1])

        box2 = self.cv2[2](x[2])
        score2 = self.cv3[2](x[2])

        return box0, score0, box1, score1, box2, score2


class ESP_Attention(Attention):
    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            -1, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(-1, C, H, W) + self.pe(
            v.reshape(-1, C, H, W)
        )
        x = self.proj(x)
        return x


class ESP_Detect_Exporter(Exporter):
    """
    adapted from ultralytics for detection task
    """

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """YOLO ONNX export."""
        requirements = ["onnx>=1.14.0"]  # from esp-ppq requirments.txt
        # since onnxslim will cause NCHW -> 1(N*C)HW in yolo11, we replace onnxslim with onnxsim
        if self.args.simplify:
            requirements += [
                "onnxsim",
                "onnxruntime" + ("-gpu" if torch.cuda.is_available() else ""),
            ]
        check_requirements(requirements)

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(
            f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}..."
        )
        f = str(self.file.with_suffix(".onnx"))
        output_names = ["box0", "score0", "box1", "score1", "box2", "score2"]
        dynamic = (
            self.args.dynamic
        )  # case 1: deploy model on ESP32, dynamic=False; case 2: QAT gt onnx for inference, dynamic=True
        if dynamic:
            dynamic = {"images": {0: "batch"}}
            for name in output_names:
                dynamic[name] = {0: "batch"}

        with arange_patch(self.args):
            torch.onnx.export(
                self.model,
                self.im,
                f,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=False,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic or None,
            )
        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.args.simplify:
            try:
                import onnxsim

                LOGGER.info(
                    f"{prefix} simplifying with onnxsim {onnxsim.__version__}..."
                )
                model_onnx, _ = onnxsim.simplify(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        onnx.save(model_onnx, f)
        return f, model_onnx


class ESP_YOLO(YOLO):
    def export(
        self,
        **kwargs,
    ):
        self._check_is_pytorch_model()
        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return ESP_Detect_Exporter(overrides=args, _callbacks=self.callbacks)(
            model=self.model
        )


# Ändere hier den Pfad zu deinem eigenen Modell
model = ESP_YOLO(r"C:\Users\rapha\Documents\GitHub\esp-dl\OWN_FILES\yolo11n_detect_train2_best.pt")
for m in model.modules():
    if isinstance(m, Attention):
        m.forward = ESP_Attention.forward.__get__(m)
    if isinstance(m, Detect):
        m.forward = ESP_Detect.forward.__get__(m)

# Passe die Parameter für ESP32S3 an
model.export(
    format="onnx", 
    simplify=True, 
    opset=13,           # Konservative Wahl für ESP32 Kompatibilität
    dynamic=False,      # Statische Eingabegröße für ESP32
    imgsz=640,          # Anpassen je nach deinen Anforderungen (320, 416, 640, etc.)
    half=False          # FP32 für bessere Kompatibilität
)
