from pathlib import Path
from typing import List

from webui.backend.cache import CacheStore


class FeedbackExporter:
    def __init__(self, cache: CacheStore):
        self.cache = cache

    def export(self, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        records = self.cache.list_feedback()
        paths = []
        for record in records:
            cache_record = self.cache.get(record["image_hash"])
            if not cache_record:
                continue
            image_path = output_dir / f"{record['image_hash']}.png"
            if not image_path.exists():
                image_path.write_bytes(cache_record["image_bytes"])
            label_path = output_dir / f"{record['image_hash']}.txt"
            label_path.write_text(record["label"])
            paths.append(image_path)
        return paths
