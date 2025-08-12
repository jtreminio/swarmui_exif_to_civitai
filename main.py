import argparse
import datetime
import json
import os
import re
import hashlib
import shutil
import math
from PIL import Image, PngImagePlugin
from concurrent.futures import ThreadPoolExecutor, as_completed


class SwarmUIExifToCivitAI:
    def __init__(self, overwrite: bool = False):
        self.overwrite = overwrite

    def process_file(self, file_path: str) -> None:
        try:
            img = Image.open(file_path)
            exif_data = img.info.get("parameters")

            if not exif_data:
                print(f"No EXIF 'parameters' field found in {file_path}.")
                return

            cleaned_exif = self._process_exif_json(exif_data)
            if not cleaned_exif:
                return

            # Generate MD5 hash of final metadata
            hash_value = hashlib.md5(cleaned_exif.encode("utf-8")).hexdigest()

            # Determine output filename
            base, ext = os.path.splitext(file_path)
            output_path = file_path if self.overwrite else f"{base}_{hash_value}{ext}"

            # Prepare metadata
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("parameters", cleaned_exif)

            # Save image
            img.save(output_path, "PNG", pnginfo=metadata)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def gather_images(self, path: str) -> list[str]:
        """
        If path is a directory, return all PNG files in it. If it's a file, return a list containing it.
        """
        if os.path.isdir(path):
            return [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".png")
            ]
        elif os.path.isfile(path):
            return [path]
        else:
            print(f"Path not found: {path}")
            return []

    def _clean_prompt(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"embedding:[^,\n]+\.safetensors", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+,", ",", text).strip()
        text = re.sub(r",+,", ", ", text)
        text = re.sub(r"\s+", " ", text).strip(" ,")

        return text

    def _process_exif_json(self, exif_data: str) -> str | None:
        try:
            exif_json = json.loads(exif_data)
            image_params = exif_json.get("sui_image_params", {})
            sui_models = exif_json.get("sui_models", [])
            if not isinstance(sui_models, list):
                print("Warning: 'sui_models' is not a list — skipping file.")
                return None

            prompt = self._clean_prompt(image_params.get("prompt", ""))
            neg_prompt = self._clean_prompt(image_params.get("negativeprompt", ""))
            steps = image_params.get("steps", 0)
            sampler = image_params.get("sampler", "")
            scheduler = image_params.get("scheduler", "")
            cfgscale = image_params.get("cfgscale", 0)
            seed = image_params.get("seed", 0)
            width = image_params.get("width", 0)
            height = image_params.get("height", 0)
            model = image_params.get("model", "")
            loraweights = image_params.get("loraweights")

            model_hash = ""
            embeddings = []
            loras = []

            for sui_model in sui_models:
                name = sui_model.get("name", "").removesuffix(".safetensors")
                ihash = sui_model.get("hash")[2:14]

                if sui_model.get("param") == "model":
                    model_hash = ihash
                    continue

                sui_model["name"] = name
                sui_model["hash"] = ihash

                if sui_model.get("param") == "used_embeddings":
                    embeddings.append(sui_model)
                    continue

                if sui_model.get("param") == "loras":
                    loras.append(sui_model)
                    continue

            for i, lora in enumerate(loras):
                lora["weight"] = float(loraweights[i])

            metadata = [
                f"Steps: {steps}",
                f"Sampler: {self._map_sampler(sampler, scheduler)}",
                f"CFG scale: {cfgscale}",
                f"Seed: {seed}",
                f"Size: {width}x{height}",
                f"Model: {model}",
                f"Model hash: {model_hash}",
            ]

            hashes = {
                "model": model_hash,
            }

            for i, lora in enumerate(loras):
                metadata.append(f"Lora_{i} Model name: {lora['name']}.safetensors")
                metadata.append(f"Lora_{i} Model hash: {lora['hash']}")
                metadata.append(f"Lora_{i} Strength model: {lora['weight']}")
                hashes[f"lora:{lora['name']}"] = lora["hash"]

            for embedding in embeddings:
                hashes[f"embed:{embedding['name']}"] = embedding["hash"]

            final_str = f"{prompt}\n" f"Negative prompt: {neg_prompt}\n"
            final_str += ", ".join(metadata) + ", Hashes: " + json.dumps(hashes)

            return final_str
        except Exception as e:
            print(f"Error processing EXIF JSON: {e}")
            return None

    def _map_sampler(self, code: str, scheduler: str) -> str:
        codes = {
            "ddim": "DDIM",
            "dpm_2": "DPM2",
            "dpm_2_ancestral": "DPM2 a",
            "dpm_adaptive": "DPM adaptive",
            "dpm_fast": "DPM fast",
            "dpmpp_2m": "DPM++ 2M",
            "dpmpp_2m_cfg_pp": "DPM++ 2M",  # CFG++
            "dpmpp_2m_sde": "DPM++ 2M SDE",
            "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",  # GPU Seeded
            "dpmpp_2s_ancestral": "DPM++ 2S a",
            "dpmpp_2s_ancestral_cfg_pp": "DPM++ 2S a",  # CFG++
            "dpmpp_3m_sde": "DPM++ 3M SDE",
            "dpmpp_3m_sde_gpu": "DPM++ 3M SDE",  # GPU Seeded
            "dpmpp_sde": "DPM++ SDE",
            "dpmpp_sde_gpu": "DPM++ SDE",  # GPU Seeded
            "euler": "Euler",
            "euler_ancestral": "Euler a",
            "euler_ancestral_cfg_pp": "Euler a",  # CFG++
            "euler_cfg_pp": "Euler",  # CFG++
            "heun": "Heun",
            "heunpp2": "Heun",  # ++ 2
            "lcm": "LCM",
            "lms": "LMS",
            "uni_pc": "UniPC",
            # unsupported in civitai
            "ddpm": "DDPM",
            "deis": "DEIS",
            "er_sde": "ER-SDE-Solver",
            "gradient_estimation": "Gradient Estimation",
            "gradient_estimation_cfg_pp": "Gradient Estimation CFG++",
            "ipndm": "iPNDM",
            "ipndm_v": "iPNDM-V",
            "res_multistep": "Res MultiStep",
            "res_multistep_ancestral": "Res MultiStep Ancestral",
            "res_multistep_ancestral_cfg_pp": "Res MultiStep Ancestral CFG++",
            "res_multistep_cfg_pp": "Res MultiStep CFG++",
            "sa_solver": "sa_solver",
            "sa_solver_pece": "sa_solver_pece",
            "seeds_2": "SEEDS 2",
            "seeds_3": "SEEDS 3",
            "uni_pc_bh2": "UniPC BH2",
        }

        if scheduler == "karras":
            codes["dpmpp_2m"] = "DPM++ 2M Karras"
            codes["dpmpp_2m_sde"] = "DPM++ 2M SDE Karras"
            codes["dpmpp_2s_ancestral"] = "DPM++ 2S a Karras"
            codes["dpmpp_3m_sde"] = "DPM++ 3M SDE Karras"
            codes["dpmpp_3m_sde_gpu"] = "DPM++ 3M SDE Karras"
            codes["dpmpp_sde"] = "DPM++ SDE Karras"
            codes["dpm_2"] = "DPM2 Karras"
            codes["dpm_2_ancestral"] = "DPM2 a Karras"
            codes["lms"] = "LMS Karras"

        if scheduler == "exponential":
            codes["dpmpp_3m_sde"] = "DPM++ 3M SDE Exponential"
            codes["dpmpp_3m_sde_gpu"] = "DPM++ 3M SDE Exponential"

        return codes.get(code, code)


def compute_folder_counts(n: int) -> list[int]:
    """
    Decide how many folders and how many images per folder.

    Rules:
    - If n <= 12: one folder.
    - Max 12 per folder.
    - Prefer ~10–11 per folder when possible.
    - Distribute as evenly as possible (sizes differ by at most 1).
    """
    if n <= 12:
        return [n]

    # Start near an average of 11 per folder
    k = max(2, round(n / 11))

    # Enforce max 12 per folder
    while math.ceil(n / k) > 12:
        k += 1

    # Prefer not to go below 10 if we can keep within max
    while k > 1 and math.ceil(n / (k - 1)) <= 12 and math.floor(n / (k - 1)) >= 10:
        k -= 1

    base = n // k
    rem = n % k
    # Result: first `rem` folders get base+1, the rest get base
    counts = [base + 1] * rem + [base] * (k - rem)
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Clean and rewrite SwarmUI EXIF metadata for CivitAI."
    )
    parser.add_argument(
        "--input",
        default="nonversioned",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    runner = SwarmUIExifToCivitAI(overwrite=True)
    files = runner.gather_images(args.input)

    if not files:
        print("No files to process.")
        return

    # Deterministic order for moving into batches
    files.sort()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(runner.process_file, f): f for f in files}
        for future in as_completed(futures):
            future.result()

    total = len(files)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ready_root = os.path.join(args.input, "../ready/")

    if total <= 12:
        # Single folder
        target_dir = os.path.join(ready_root, timestamp)
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            shutil.move(f, target_dir)
        print(f"Moved {total} file(s) into: {target_dir}")
        return

    # Multiple folders: spread evenly, prefer ~10–11, cap at 12
    counts = compute_folder_counts(total)
    print(f"Distributing {total} files across {len(counts)} folder(s): {counts}")

    start = 0
    for idx, count in enumerate(counts, start=1):
        batch_files = files[start:start + count]
        start += count
        batch_dir = os.path.join(ready_root, f"{timestamp}-{idx:02d}")
        os.makedirs(batch_dir, exist_ok=True)
        for f in batch_files:
            shutil.move(f, batch_dir)
        print(f"Moved {len(batch_files)} file(s) into: {batch_dir}")


if __name__ == "__main__":
    main()
