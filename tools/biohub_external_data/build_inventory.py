"""Compile a portable, aggregate inventory from the saved download receipts.

Writes ignored work/ output; copying an inventory into docs is an explicit
documentation action. This does not download, train, or rehash all raw payloads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "work/biohub-forum-archive"
GUIDE = REPO / "work/biohub-data-guide"
OUTPUT = GUIDE / "branch_inventory"
TOPIC = "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103"


# Source identities recorded from the captured Zoo species pages.
ZOO_SOURCE_METADATA = {
    "ascidian": {
        "species": "Phallusia mammillata",
        "source_study_as_archived": "Guignard, Science 2020; original-author Astec tracking",
        "study_links": [
            "https://www.science.org/doi/10.1126/science.aar5663"
        ],
        "optional_original_image_links_not_downloaded": [
            "https://figshare.com/s/765d4361d1b073beedd5#/articles/8223890"
        ]
    },
    "drosophila": {
        "species": "Drosophila melanogaster",
        "source_study_as_archived": "Amat, Nature Methods 2014; original-author GMM tracking",
        "study_links": [
            "https://doi.org/10.1038/nmeth.3036"
        ],
        "optional_original_image_links_not_downloaded": []
    },
    "elegans": {
        "species": "Caenorhabditis elegans",
        "source_study_as_archived": "Moyle, Nature 2021; original-author Linajea tracking",
        "study_links": [
            "https://doi.org/10.1038/s41586-020-03169-5"
        ],
        "optional_original_image_links_not_downloaded": [
            "https://zenodo.org/records/6460375"
        ]
    },
    "mouse": {
        "species": "Mus musculus",
        "source_study_as_archived": "McDole, Cell 2018; original-author tracking",
        "study_links": [
            "https://www.cell.com/cell/fulltext/S0092-8674(18)31243-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418312431%3Fshowall%3Dtrue"
        ],
        "optional_original_image_links_not_downloaded": [
            "https://idr.openmicroscopy.org/webclient/?show=project-502"
        ]
    },
    "tribolium": {
        "species": "Tribolium castaneum",
        "source_study_as_archived": "Akanksha Jain; Ultrack tracking",
        "study_links": [],
        "optional_original_image_links_not_downloaded": [
            "https://celltrackingchallenge.net/3d-datasets/"
        ]
    },
    "zebrafish": {
        "species": "Danio rerio",
        "source_study_as_archived": "Lange/Zebrahub; original-author Ultrack tracking",
        "study_links": [
            "https://www.biorxiv.org/content/10.1101/2023.03.06.531398v2"
        ],
        "optional_original_image_links_not_downloaded": []
    }
}

def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(REPO))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def receipt(path):
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    original = read(ARCHIVE / "verification.json")
    verified = original["verified"]
    external = read(ARCHIVE / "external-downloads.json")
    zoo_stats = read(GUIDE / "prepared/zoo_manifest.json")
    synthetic_stats = read(GUIDE / "prepared/synthetic_audit.json")
    riken_stats = read(GUIDE / "prepared/riken_audit.json")
    prepared_receipt = read(GUIDE / "verification.json")
    prepared_hashes = {x["path"]: x for x in read(GUIDE / "prepared_checksums.json")}

    def file_record(name):
        saved = verified[name]
        return {"path": relative(ARCHIVE / name),
                **{k: saved[k] for k in ("bytes", "sha256", "archive_members", "zip_crc", "contents") if k in saved}}

    def prepared_record(name):
        r = prepared_hashes[name]
        return {"path": relative(GUIDE / name), "bytes": r["bytes"], "sha256": r["sha256"]}

    def tree(prefix):
        rows = sorted([name.removeprefix(prefix), value["bytes"], value["sha256"]]
                      for name, value in verified.items() if name.startswith(prefix))
        encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
        return {"path": relative(ARCHIVE / prefix), "files": len(rows),
                "bytes": sum(r[1] for r in rows),
                "recorded_content_tree_sha256": hashlib.sha256(encoded).hexdigest()}

    synthetic_root = ARCHIVE / "downloads/kaggle/biohub_synthetic"
    synthetic_metadata = read(synthetic_root / "metadata.json")
    synthetic = {
        "id": "biohub_synthetic", "kind": "simulated microscopy and dense centre/graph labels",
        "source_url": "https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset",
        "release_identity": "2026-09-08 captured output; no immutable Kaggle version number recorded",
        "license": {"declaration": "CC0", "source_url": TOPIC, "scope": "author's dataset declaration"},
        "native_spacing_um_zyx": [1.625, .40625, .40625],
        "sampled_spacing_um_zyx": [1.625, 1.625, 1.625],
        "collections": [
            {"id": "static", **tree("downloads/kaggle/biohub_synthetic/static/"),
             "samples": 1539, "centre_labels": synthetic_stats["static_centroids"],
             "image": {"key": "volume", "dtype": "uint16", "axes": "ZYX", "shape": [64, 256, 256]},
             "source_labels": {"centroids": {"shape": ["N", 3], "dtype": "float32", "axes": "ZYX", "units": "native voxels"}},
             "temporal_labels": False,
             "prepared_path_pattern": "work/biohub-data-guide/prepared/synthetic/vol_XXXXX_labels.npz"},
            {"id": "sequences", **tree("downloads/kaggle/biohub_synthetic/sequences/"),
             "samples": 2174, "node_observations": synthetic_stats["nodes"],
             "edges": synthetic_stats["edges"], "division_parents": synthetic_stats["division_parents"],
             "image": {"key": "volumes", "dtype": "uint16", "axes": "TZYX", "shape": [6, 64, 64, 64]},
             "source_labels": {
                 "nodes": {"dtype": "float32", "columns": ["t", "z", "y", "x", "track_id"],
                           "coordinate_units": "native voxels despite already-sampled images",
                           "track_id_semantics": "clone/lineage ID inherited by daughters"},
                 "edges": {"dtype": "int32", "shape": ["E", 2], "semantics": "source/target node-row indices"},
                 "divisions": {"dtype": "int32", "semantics": "parent node-row indices"}},
             "coordinate_conversion_to_images": "z unchanged; y/4; x/4",
             "native_sequence_images_available": False,
             "prepared_path_pattern": "work/biohub-data-guide/prepared/synthetic/seq_XXXX_labels.npz"}
        ],
        "dense_segmentation_masks": False,
        "prepared_manifest_path": "work/biohub-data-guide/prepared/synthetic_manifest.json",
        "prepared_common_fields": {
            "zyx_native": "float32 N×3, native voxels",
            "zyx_pooled": "float32 N×3, Z unchanged and Y/X divided by four",
            "zyx_um": "float32 N×3, physical micrometres"},
        "prepared_sequence_fields": {
            "node_id": "int64, dense file-local row ID", "t": "int64 frame index",
            "edges": "int64 E×2 node-row indices",
            "division_parent_ids": "int64 parent node IDs",
            "tracklet_id": "int64 continuous-path segment; splits after a fork",
            "source_clone_id": "int64 preserved source lineage ID"},
        "source_clone_time_repeated_rows": synthetic_stats["clone_time_collisions"],
        "prepared_split_counts": {"synthetic_train": 3359, "synthetic_holdout": 354},
        "split_scope": "deterministic whole-example generator sanity split; not biological validation",
        "censoring": "last-frame future unavailable; loader division_target_observed masks those rows",
        "calibration": {"density_and_tissue_shape": "first ten videos of first sorted embryo, 44b6 in the captured dataset",
                        "other_motion_and_render_parameters": "fixed/real-informed; complete fold provenance not established"},
        "captured_generator_metadata": {
            "real_detected_count_range": synthetic_metadata["real_count_range"],
            "shell_kind": synthetic_metadata["shell_kind"],
            "claimed_motion_calibration": synthetic_metadata["motion_calibration"],
            "configuration": synthetic_metadata["config"],
            "scope": "author metadata; not a local revalidation of physical motion or real division prevalence"},
        "generator_source_observations": {
            "source_function": "run_dataset_build in captured builder notebook Python export",
            "seed_schedule": {"static": "100000+i", "sequence_motion": "500000+j", "sequence_frame_render": "900000+j*97+t"},
            "initial_count": "max(20, choice(real_detected_counts)*1.25), before placement exclusions",
            "default_division_probability_per_transition": 0.05,
            "continuation_motion": "velocity is added directly to native voxel ZYX, although calibration constants are described as micrometres/frame",
            "sister_offset": "half-separation converted from micrometres by division by native spacing",
            "post_transition_clipping_native_zyx": {"lower": [4, 12, 12], "upper": [59, 243, 243]},
            "births_deaths": "existing cells continue or split; no explicit death, independent entry or exit process",
            "rendering": "each frame uses a different renderer seed and resamples radii for supplied centers; original persistent appearance labels are absent"},
        "division_frequency": "deliberately oversampled; 165267/4056226 is descriptive, not real prevalence",
        "metadata_receipts": [receipt(synthetic_root / n) for n in ["manifest.json", "metadata.json", "progress.log"]],
        "inspection_examples": [file_record("downloads/kaggle/biohub_synthetic/" + n)
                                for n in ["static/vol_00000.npz", "sequences/seq_0000.npz"]],
        "validation": "all 3713 prepared label sets passed recorded coordinate and graph checks"
    }

    zoo = []
    for stats in zoo_stats:
        species = Path(stats["prepared"]).stem.removesuffix("_graph")
        job = next(x for x in external if "/tracks_" + species + "_" in x["path"])
        base_url = job["url"].rsplit("/", 1)[0] + "/"
        base_file = f"downloads/virtual-embryo-zoo/tracks_{species}_bundle.zarr.zip"
        directory = f"downloads/virtual-embryo-zoo/tracks_{species}_attributes_bundle.zarr"
        group = ARCHIVE / directory
        arrays = {}
        for key in ["points", "tracks_to_points/data", "tracks_to_points/indices", "tracks_to_points/indptr",
                    "tracks_to_tracks/data", "tracks_to_tracks/indices", "tracks_to_tracks/indptr"]:
            meta = read(group / key / ".zarray")
            arrays[key] = {k: meta[k] for k in ["shape", "dtype", "chunks", "zarr_format"]}
        record = {
            "id": "zoo_" + species, "species_key": species,
            **ZOO_SOURCE_METADATA[species],
            "optional_image_link_scope": "recorded page links, not downloaded or audited for correspondence",
            "kind": "experimental tracking output; no downloaded microscopy",
            "source_page": job["source"],
            "base_bundle": {"url": base_url + Path(base_file).name, **file_record(base_file)},
            "enriched_store": {"url": base_url + Path(directory).name + "/", **tree(directory + "/")},
            "prepared_graph": prepared_record(stats["prepared"]),
            "counts": {k: stats[k] for k in ["frames", "nodes", "tracklets", "edges", "division_parents",
                                             "within_track_gaps", "nonadjacent_parent_links"]},
            "count_semantics": "nodes are frame-specific observations, not unique biological cells",
            "physical_scale_known": False, "physical_frame_duration_known": False,
            "coordinates": "raw source ZYX; no guessed inverse display normalization applied",
            "point_fields": stats["coordinate_fields"], "source_arrays": arrays,
            "viewer_attributes": read(group / "attributes" / ".zattrs"),
            "prepared_fields": {
                "node_id": "int64 file-local node row", "source_point_id": "int64 t*max_points_per_frame+slot",
                "t": "int64 source frame index", "zyx_source": "float32 N×3 raw source coordinates",
                "tracklet_id": "int64 per-node segment ID", "parent_tracklet": "int64 per-track parent ID, root -1",
                "edges": "int64 E×2 adjacent-frame node-row indices",
                "division_parent_ids": "int64 parent node rows derived from reconstructed graph"},
            "label_quality": "graph integrity checked; not assumed manually verified biological truth",
            "images_downloaded": False, "masks_downloaded": False,
            "duplicate_representations": "base bundle and enriched store overlap",
            "license_scope": "originating study terms; viewer software MIT is not a blanket data license",
        }
        if species == "zebrafish":
            record["organizer_statement"] = {
                "date": "2026-08-13", "author_role": "HOST", "author": "Thibgolds",
                "source_url": "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330",
                "paraphrase": "Public Zebrahub resources are allowed; host reports no hidden-test overlap",
                "scope_limit": "does not establish physical scale or tracking accuracy of this export"}
        if species == "mouse":
            record["division_limit"] = "no recorded forks; absence is not biological nondivision ground truth"
        if species == "tribolium":
            rebuilt = directory + ".zip"
            record["enriched_zip"] = {
                "advertised_url": base_url + Path(rebuilt).name,
                "upstream_result_on_download_date": "HTTP 404",
                "local_replacement": file_record(rebuilt),
                "provenance": "rebuilt from all 1151 publicly served folder files; not upstream ZIP bytes"}
        zoo.append(record)

    riken = []
    for job in external:
        if "/ssbd/" not in job["path"]:
            continue
        name = Path(job["path"]).name.removeprefix("zebrafish_").removesuffix("_bdml3.0.zip")
        record = {"id": "riken_" + name, "source_page": job["source"],
                  "download_url": job["url"], **file_record(job["path"]),
                  "published_sha256": job["sha256"],
                  "published_hash_matched": job["sha256"] == verified[job["path"]]["sha256"],
                  "images_downloaded": False, "masks_downloaded": False,
                  "license": "CC BY-NC-SA as stated on captured project page",
                  "independence_of_views_or_specimens": "not established from archive names",
                  "inspection_scope": "file inventory, ZIP integrity and published checksum only",
                  "label_schema_verified": False}
        if name == "animal_c":
            record.update({
                "inspection_scope": "full HDF5 structure/count audit; first ten frames exported",
                "label_schema_verified": True, "frames": riken_stats["frames"],
                "point_measurements": riken_stats["point_measurements"],
                "physical_coordinate_unit": "micrometre", "time_step_seconds": 90,
                "explicit_temporal_edges_found": False,
                "id_semantics": "frame-specific measurement ID; persistent identity not established",
                "object_schema": {"path": "data/<numeric-frame>/object/0",
                                  "fields": ["ID", "t", "entity", "x", "y", "z"],
                                  "encoding": "structured fields, numerical values may be byte strings"},
                "feature_schema": {"path": "data/<numeric-frame>/feature/0", "fields": ["ID", "fID", "value"]},
                "feature_definitions": riken_stats["feature_definitions"],
                "feature_limits": "missing and unusually broad FWHM extents; no recovered pixel masks",
                "prepared_preview": prepared_record("prepared/riken_animal_c_first10_frames.npz"),
                "preview_points": 1312, "preview_frames": 10,
                "prepared_fields": {"t": "int64 0–9", "zyx_um": "float32 N×3 micrometres",
                                    "measurement_id": "Unicode source measurement ID"},
                "time_index_mapping": "display t=0 maps to HDF5 object time index 1; elapsed time relative to excerpt start"})
        riken.append(record)

    notebooks = []
    for path in sorted((ARCHIVE / "notebooks").rglob("kernel-metadata.json")):
        meta = read(path)
        notebooks.append({k: meta[k] for k in ["id", "id_no", "title", "language", "competition_sources", "kernel_sources",
                                             "dataset_sources", "model_sources", "docker_image"] if k in meta}
                         | {"url": "https://www.kaggle.com/code/" + meta["id"],
                            "immutable_version_recorded": None,
                            "executed_during_download": False,
                            "artifacts": [receipt(p) for p in sorted(path.parent.iterdir()) if p.is_file()]})

    extra_csvs = [{**job, **file_record(job["path"])}
                 for job in read(ARCHIVE / "zoo-additional-files.json")]
    for item in extra_csvs:
        with (REPO / item["path"]).open() as source:
            item["header"] = source.readline().strip().split(",")
        item["duplicate_of"] = "ascidian enriched tracking store" if "ascidian" in item["path"] else "elegans enriched tracking store"
        item["validation"] = "exact time, coordinate and parent relationship match after ID remapping"

    inventory = {
        "schema_version": 1, "documented_on": "2026-09-09", "data_snapshot": "2026-09-08",
        "purpose": "factual information for a downstream decision; no dataset or training strategy selected",
        "discussion": {"url": TOPIC, "retrieved_messages": 11, "reported_replies": 10, "unavailable_messages_detected": 0,
                       "scope": "original post and ten replies in this topic; not the entire competition forum",
                       "original_message_id": 3507381,
                       "reply_message_ids": [3521851, 3521291, 3521235, 3521140, 3521070, 3521156,
                                             3510364, 3510754, 3509286, 3509458]},
        "availability": {"included_in_git": "this information and authored code; not original datasets or prepared arrays",
                         "historical_local_roots": [relative(ARCHIVE), relative(GUIDE)],
                         "canonical_competition_input": "/kaggle/input/competitions/biohub-cell-tracking-during-development",
                         "fresh_clone": "source files and generated arrays require separate acquisition/preparation",
                         "source_paths": "repository-relative unless canonical input explicitly stated"},
        "fingerprint_definition": "SHA256 of UTF-8 JSON sorted [relative_path_within_tree,bytes,sha256] rows, separators=(',',':'), ensure_ascii=True; uses recorded file hashes, not a fresh raw rehash",
        "download_totals": {k: original[k] for k in ["expected_files", "verified_files", "verified_bytes", "incomplete", "errors"]},
        "download_total_scope": "storage includes overlapping base/enriched Zoo exports and rebuilt Tribolium ZIP; not unique training volume",
        "synthetic": synthetic, "zoo": zoo, "riken": riken, "additional_csvs": extra_csvs,
        "forum_illustration": {**file_record("attachments/Selection_4780.png"), "training_dataset": False,
                               "source_message_id": 3521291},
        "notebooks": notebooks,
        "uninspected_or_unavailable": [
            "No paired image volumes downloaded for Zoo or RIKEN.",
            "Six RIKEN archives have only inventory and integrity checks, not the animal C label audit.",
            "Zoo physical scales, physical frame cadence, biological label quality and cross-export deduplication beyond named representations remain unverified.",
            "Exact upstream enriched Tribolium ZIP was unavailable; public file contents were recovered.",
            "No immutable Kaggle notebook output version was captured; source and payload fingerprints describe the local snapshot."
        ],
        "cross_source_biological_overlap": "not audited beyond base/enriched/CSV representation overlap; acquisition identity must not be inferred solely from filename or species",
        "reference_pins": {"metric": "075fc5f5a52d11077f9dc2b074644618f26939e2",
                           "intracktive_converter": "8afc30b5c0a42632ad971e76dbdb559286bec0bf"},
        "evidence_files": [receipt(p) for p in [ARCHIVE / "verification.json", GUIDE / "verification.json",
                            GUIDE / "prepared_checksums.json", GUIDE / "prepared/synthetic_audit.json",
                            GUIDE / "prepared/zoo_manifest.json", GUIDE / "prepared/riken_audit.json"]]
    }
    verification = {"schema_version": 1, "documented_on": "2026-09-09",
                    "source_receipt_scope": "saved September 8 data snapshot; aggregate facts copied without raw per-file payloads",
                    "download": inventory["download_totals"],
                    "preparation": prepared_receipt,
                    "visual_examples": read(GUIDE / "checks/visual-examples-verification.json"),
                    "trajectories": read(GUIDE / "checks/trajectory-verification.json"),
                    "browser": read(GUIDE / "checks/browser-verification.json"),
                    "training_or_score_gain_measured_with_external_data": False}
    motion_audit = GUIDE / "checks/synthetic-motion-audit.json"
    if motion_audit.exists():
        verification["synthetic_motion_audit"] = read(motion_audit)
    assert len(zoo) == 6 and len(riken) == 7 and len(extra_csvs) == 2
    assert all(x["published_hash_matched"] for x in riken)
    assert synthetic["collections"][0]["files"] == 1539
    assert synthetic["collections"][1]["files"] == 2174
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, content in [("dataset_inventory.json", inventory), ("verification_summary.json", verification)]:
        (OUTPUT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(relative(OUTPUT / name))


if __name__ == "__main__":
    main()
