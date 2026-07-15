"""COCO/VOC image loading and deterministic subset selection."""

from __future__ import annotations

import os
import random
from pathlib import Path
from xml.etree import ElementTree

import cv2
from pycocotools.coco import COCO

from .utils.image_selection import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageSelection,
    collect_image_paths,
    select_image_paths,
)

from .types import ImageSample


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_CLASS_ALIASES = {
    "airplane": "aeroplane",
    "dining table": "diningtable",
    "dining_table": "diningtable",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "potted_plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
}


class CocoDetectionDataset:
    """Load COCO-style images as RGB uint8 arrays.

    `num_images` is resolved through `utils.image_selection`, so integer values
    use a deterministic random subset controlled by `sample_seed`.
    """

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        num_images: str = "all",
        sample_seed: int = 2,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.ann_file = Path(ann_file)
        self.coco = COCO(str(self.ann_file))
        self.selection = select_image_paths(
            self.image_dir,
            num_images=num_images,
            sample_strategy="random",
            sample_seed=sample_seed,
        )
        self.img_ids = self._selection_to_img_ids(self.selection)

    def _selection_to_img_ids(self, selection: ImageSelection) -> list[int]:
        id_by_file = {
            Path(info["file_name"]).as_posix(): int(info["id"])
            for info in self.coco.loadImgs(self.coco.getImgIds())
        }
        id_by_name = {
            Path(info["file_name"]).name: int(info["id"])
            for info in self.coco.loadImgs(self.coco.getImgIds())
        }

        img_ids = []
        for image_path in selection.selected_image_paths:
            rel = Path(os.path.relpath(image_path, self.image_dir)).as_posix()
            if rel in id_by_file:
                img_ids.append(id_by_file[rel])
                continue

            name = Path(image_path).name
            if name in id_by_name:
                img_ids.append(id_by_name[name])
                continue

            raise KeyError(f"Selected image is not present in COCO annotations: {image_path}")
        return img_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def __iter__(self):
        for img_id in self.img_ids:
            yield self.load_sample(img_id)

    def load_sample(self, img_id: int) -> ImageSample:
        info = self.coco.loadImgs([img_id])[0]
        path = self.image_dir / info["file_name"]
        if not path.is_file():
            path = self.image_dir / Path(info["file_name"]).name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageSample(
            image_id=int(img_id),
            file_name=info["file_name"],
            path=path,
            image=rgb,
        )

    def load_samples(self) -> list[ImageSample]:
        return [self.load_sample(img_id) for img_id in self.img_ids]

    def label_to_category_id(self, class_names: list[str] | tuple[str, ...]) -> dict[int, int]:
        cat_ids = sorted(self.coco.getCatIds())
        cat_by_name = {cat["name"]: cat["id"] for cat in self.coco.loadCats(cat_ids)}
        mapper: dict[int, int] = {}

        for label, name in enumerate(class_names):
            mapper[label] = int(cat_by_name.get(name, cat_ids[label]))
        return mapper


class VocDetectionDataset:
    """Load a Pascal VOC split and expose a COCO-compatible evaluation API.

    ``ann_file`` is a VOC image-set file such as ``ImageSets/Main/val.txt``.
    XML files are read from the sibling ``Annotations`` directory by default.
    Only IDs listed in the image-set file participate in image selection.
    """

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        num_images: str = "all",
        sample_seed: int = 2,
        annotation_dir: str | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.ann_file = Path(ann_file)
        self.annotation_dir = (
            Path(annotation_dir)
            if annotation_dir is not None
            else self.image_dir.parent / "Annotations"
        )

        image_ids = self._read_image_set()
        image_paths = self._resolve_image_paths(image_ids)
        self.selection = self._select_image_paths(
            image_paths,
            num_images=num_images,
            sample_seed=sample_seed,
        )

        voc_id_by_path = {
            os.path.normpath(str(path)): image_id
            for image_id, path in zip(image_ids, image_paths)
        }
        selected_voc_ids = [
            voc_id_by_path[os.path.normpath(path)]
            for path in self.selection.selected_image_paths
        ]
        coco_id_by_voc_id = {
            image_id: index
            for index, image_id in enumerate(image_ids, start=1)
        }
        self.img_ids = [coco_id_by_voc_id[image_id] for image_id in selected_voc_ids]
        self.coco = self._build_coco_api(selected_voc_ids, coco_id_by_voc_id)

    def _read_image_set(self) -> list[str]:
        if not self.ann_file.is_file():
            raise FileNotFoundError(f"VOC image-set file does not exist: {self.ann_file}")

        image_ids = []
        seen = set()
        for line in self.ann_file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            image_id = fields[0]
            if image_id not in seen:
                image_ids.append(image_id)
                seen.add(image_id)
        if not image_ids:
            raise ValueError(f"VOC image-set file is empty: {self.ann_file}")
        return image_ids

    def _resolve_image_paths(self, image_ids: list[str]) -> list[Path]:
        available = {
            Path(path).stem: Path(path)
            for path in collect_image_paths(self.image_dir)
            if Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        missing = [image_id for image_id in image_ids if image_id not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"VOC image-set references {len(missing)} missing image(s) in "
                f"{self.image_dir}: {preview}"
            )
        return [available[image_id] for image_id in image_ids]

    def _select_image_paths(
        self,
        image_paths: list[Path],
        num_images: str | int | None,
        sample_seed: int,
    ) -> ImageSelection:
        normalized = sorted(os.path.normpath(str(path)) for path in image_paths)
        if num_images is None or str(num_images).lower() == "all":
            limit = None
        else:
            try:
                limit = int(num_images)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Invalid --num-images value {num_images!r}; use "all" or a positive integer.'
                ) from exc
            if limit < 1:
                raise ValueError(f"--num-images must be positive, got {limit}.")
            if limit > len(normalized):
                raise ValueError(
                    f"--num-images {limit} exceeds available VOC split images "
                    f"({len(normalized)})."
                )

        if limit is None:
            selected = list(normalized)
        else:
            selected = random.Random(sample_seed).sample(normalized, limit)
            selected.sort()

        return ImageSelection(
            image_dir=os.path.normpath(str(self.image_dir)),
            num_images="all" if limit is None else str(limit),
            sample_strategy="random",
            sample_seed=sample_seed,
            all_image_paths=normalized,
            selected_image_paths=selected,
        )

    def _build_coco_api(
        self,
        selected_voc_ids: list[str],
        coco_id_by_voc_id: dict[str, int],
    ) -> COCO:
        category_id_by_name = {
            name: index for index, name in enumerate(VOC_CLASSES, start=1)
        }
        images = []
        annotations = []
        annotation_id = 1
        selected_image_path_by_voc_id = {
            Path(path).stem: Path(path)
            for path in self.selection.selected_image_paths
        }

        for voc_id in selected_voc_ids:
            xml_path = self.annotation_dir / f"{voc_id}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(f"VOC annotation does not exist: {xml_path}")
            root = ElementTree.parse(xml_path).getroot()
            size = root.find("size")
            if size is None:
                raise ValueError(f"VOC annotation has no <size>: {xml_path}")
            width = int(size.findtext("width", "0"))
            height = int(size.findtext("height", "0"))
            image_id = coco_id_by_voc_id[voc_id]
            image_path = selected_image_path_by_voc_id[voc_id]
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "voc_id": voc_id,
                }
            )

            for obj in root.findall("object"):
                class_name = obj.findtext("name", "").strip().lower()
                if class_name not in category_id_by_name:
                    raise ValueError(
                        f"Unknown Pascal VOC class {class_name!r} in {xml_path}"
                    )
                bbox = obj.find("bndbox")
                if bbox is None:
                    raise ValueError(f"VOC object has no <bndbox>: {xml_path}")
                x1 = float(bbox.findtext("xmin", "0")) - 1.0
                y1 = float(bbox.findtext("ymin", "0")) - 1.0
                x2 = float(bbox.findtext("xmax", "0"))
                y2 = float(bbox.findtext("ymax", "0"))
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                difficult = int(obj.findtext("difficult", "0"))
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id_by_name[class_name],
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": difficult,
                        "ignore": difficult,
                    }
                )
                annotation_id += 1

        coco = COCO()
        coco.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": category_id, "name": name, "supercategory": "object"}
                for name, category_id in category_id_by_name.items()
            ],
        }
        coco.createIndex()
        return coco

    def __len__(self) -> int:
        return len(self.img_ids)

    def __iter__(self):
        for img_id in self.img_ids:
            yield self.load_sample(img_id)

    def load_sample(self, img_id: int) -> ImageSample:
        info = self.coco.loadImgs([img_id])[0]
        path = self.image_dir / info["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageSample(
            image_id=int(img_id),
            file_name=info["file_name"],
            path=path,
            image=rgb,
        )

    def load_samples(self) -> list[ImageSample]:
        return [self.load_sample(img_id) for img_id in self.img_ids]

    def label_to_category_id(self, class_names: list[str] | tuple[str, ...]) -> dict[int, int]:
        category_id_by_name = {
            name: index for index, name in enumerate(VOC_CLASSES, start=1)
        }
        mapper = {}
        for label, name in enumerate(class_names):
            normalized = str(name).strip().lower()
            normalized = VOC_CLASS_ALIASES.get(normalized, normalized)
            if normalized in category_id_by_name:
                mapper[label] = category_id_by_name[normalized]
        return mapper


def build_detection_dataset(
    dataset_format: str,
    image_dir: str,
    ann_file: str | None,
    num_images: str = "all",
    sample_seed: int = 2,
    voc_annotation_dir: str | None = None,
) -> CocoDetectionDataset | VocDetectionDataset:
    """Build the dataset adapter selected by the detection attack config."""
    if not ann_file:
        raise ValueError(
            f"Detection mode requires --ann-file for {dataset_format.upper()} data."
        )

    normalized_format = dataset_format.strip().lower()
    if normalized_format == "coco":
        return CocoDetectionDataset(
            image_dir=image_dir,
            ann_file=ann_file,
            num_images=num_images,
            sample_seed=sample_seed,
        )
    if normalized_format == "voc":
        return VocDetectionDataset(
            image_dir=image_dir,
            ann_file=ann_file,
            num_images=num_images,
            sample_seed=sample_seed,
            annotation_dir=voc_annotation_dir,
        )
    raise ValueError("dataset_format must be 'coco' or 'voc'.")
