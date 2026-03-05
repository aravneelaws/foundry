"""Synthetic dataset for RF3 training benchmarks.

Generates random tensors matching the exact shapes and dtypes that the RF3 training
pipeline expects. This allows benchmarking training throughput without any external
data dependencies (no PDB mirror, no parquet files, no MSAs).

The returned examples are structurally valid:
- atom_to_token_map is a valid contiguous mapping
- ref_space_uid groups atoms by token
- Diffusion timesteps follow the EDM noise schedule range
- All tensor shapes match production training dimensions
"""

import torch
from torch.utils.data import Dataset


class SyntheticRF3Dataset(Dataset):
    """Generates random tensors matching RF3 training input shapes for benchmarking.

    Args:
        num_examples: Number of synthetic examples in the dataset.
        num_tokens: Number of tokens (I) per example. Default 384 (crop_size from base.yaml).
        num_atoms: Number of atoms (L) per example. Default 3072 (~8 atoms/token average).
        diffusion_batch_size: Number of diffusion samples (D) per example. Default 48.
        num_msa_sequences: Number of MSA sequences (S) per example. Default 1024.
        n_recycles: Number of recycles for MSA stack dimension. Default 4.
        seed: Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        num_examples: int = 1000,
        num_tokens: int = 384,
        num_atoms: int = 3072,
        diffusion_batch_size: int = 48,
        num_msa_sequences: int = 1024,
        n_recycles: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.num_tokens = num_tokens  # I
        self.num_atoms = num_atoms  # L
        self.diffusion_batch_size = diffusion_batch_size  # D
        self.num_msa_sequences = num_msa_sequences  # S
        self.n_recycles = n_recycles
        self.seed = seed

        # Pre-compute the atom-to-token mapping (shared across all examples)
        self._atom_to_token_map = self._build_atom_to_token_map()
        self._ref_space_uid = self._build_ref_space_uid()

    def _build_atom_to_token_map(self) -> torch.Tensor:
        """Build a valid atom-to-token mapping where each token appears at least once."""
        I, L = self.num_tokens, self.num_atoms
        # Ensure each token 0..I-1 appears at least once
        # First I atoms get one token each, remaining atoms distributed evenly
        base = torch.arange(I)  # [I] -- one atom per token
        remaining = L - I
        if remaining > 0:
            # Distribute remaining atoms across tokens roughly evenly
            extra = torch.arange(remaining) % I
            mapping = torch.cat([base, extra.sort().values])
        else:
            mapping = base[:L]
        return mapping.long()

    def _build_ref_space_uid(self) -> torch.Tensor:
        """Build ref_space_uid that groups atoms by their token assignment."""
        # ref_space_uid = atom_to_token_map (atoms belonging to same token share same uid)
        return self._atom_to_token_map.clone()

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict:
        """Generate a single synthetic training example.

        Returns a dict matching the exact structure expected by RF3Trainer.training_step().
        """
        # Use a deterministic seed per example for reproducibility
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)

        I = self.num_tokens
        L = self.num_atoms
        D = self.diffusion_batch_size
        S = self.num_msa_sequences

        # ---- Features dict (input["f"]) ----
        feats = {
            # Atom-level features
            "atom_to_token_map": self._atom_to_token_map.clone(),  # [L]
            "ref_pos": torch.randn(L, 3, generator=gen),  # [L, 3]
            "ref_charge": torch.randn(L, generator=gen),  # [L]
            "ref_mask": torch.ones(L, dtype=torch.bool),  # [L] all valid
            "ref_element": torch.zeros(L, 128),  # [L, 128] one-hot
            "ref_atom_name_chars": torch.zeros(L, 4, 64),  # [L, 4, 64] one-hot
            "ref_space_uid": self._ref_space_uid.clone(),  # [L]
            "ref_pos_ground_truth": torch.randn(L, 3, generator=gen),  # [L, 3]
            "has_atom_level_embedding": torch.zeros(
                L, 1
            ),  # [L, 1] -- no atom-level embeddings in synthetic data
            # Atom-level embeddings: [n_conformers, L, embedding_dim]
            # Required by model when use_atom_level_embedding=True
            "atom_level_embedding": torch.randn(8, L, 384, generator=gen),
            # Token-level features
            "restype": torch.zeros(I, 32),  # [I, 32] one-hot residue type
            "profile": torch.zeros(I, 32),  # [I, 32] MSA profile
            "deletion_mean": torch.zeros(I),  # [I]
            "token_bonds": torch.zeros(I, I),  # [I, I] bond adjacency
            "asym_id": torch.zeros(I, dtype=torch.long),  # [I] single chain
            "residue_index": torch.arange(I, dtype=torch.long),  # [I]
            "entity_id": torch.zeros(I, dtype=torch.long),  # [I] single entity
            "sym_id": torch.zeros(I, dtype=torch.long),  # [I]
            "token_index": torch.arange(I, dtype=torch.long),  # [I]
            "cyclic_asym_ids": [],  # empty for synthetic
            # MSA features: [n_recycles, S, I, dim_raw_msa]
            # dim_raw_msa = 35 in rf3.yaml override (34 in base rf3_net.yaml)
            "msa_stack": torch.randn(self.n_recycles, S, I, 35, generator=gen),
            # Template / distogram conditioning
            "has_distogram_condition": torch.zeros(I, I, dtype=torch.bool),  # [I, I]
            "distogram_condition_noise_scale": torch.zeros(I),  # [I]
            "distogram_condition": torch.zeros(I, I, 64),  # [I, I, 64]
            # Ligand/nucleic flags
            "is_ligand": torch.zeros(I, dtype=torch.bool),  # [I]
            "is_dna": torch.zeros(I, dtype=torch.bool),  # [I]
            "is_rna": torch.zeros(I, dtype=torch.bool),  # [I]
            # Chiral features (empty tensors -- no chirality in synthetic data)
            "chiral_centers": torch.zeros(0, 4, dtype=torch.long),
            "chiral_center_dihedral_angles": torch.zeros(0),
        }

        # Set one-hot element encoding (carbon = index 6 for all atoms)
        feats["ref_element"][:, 6] = 1.0

        # Set one-hot restype (alanine = index 0 for all tokens)
        feats["restype"][:, 0] = 1.0

        # Set one-hot atom name chars (just 'C' for first char, 'A' for second)
        feats["ref_atom_name_chars"][:, 0, ord("C") - ord("A")] = 1.0
        feats["ref_atom_name_chars"][:, 1, 0] = 1.0  # 'A'

        # ---- Diffusion tensors ----
        # Sample timesteps from EDM noise schedule range [s_min=4e-4, s_max=160]
        # Using log-uniform distribution matching SampleEDMNoise
        log_t = torch.empty(D).uniform_(
            torch.tensor(4e-4).log().item(),
            torch.tensor(160.0).log().item(),
            generator=gen,
        )
        t = log_t.exp()  # [D]

        # Noise scaled by sigma_data=16
        noise = torch.randn(D, L, 3, generator=gen) * 16.0  # [D, L, 3]

        # Coordinates to be noised (ground truth repeated D times)
        coord_gt = torch.randn(L, 3, generator=gen)  # [L, 3]
        coord_atom_lvl_to_be_noised = (
            coord_gt.unsqueeze(0).expand(D, -1, -1).clone()
        )  # [D, L, 3]

        # ---- Ground truth ----
        ground_truth = {
            "coord_atom_lvl": coord_gt.clone(),  # [L, 3]
            "mask_atom_lvl": torch.ones(L, dtype=torch.bool),  # [L]
            # Token-level ground truth (representative atom per token)
            # Required by DistogramLoss -- use first I atom coords as token representatives
            "coord_token_lvl": coord_gt[:I].clone(),  # [I, 3]
            "mask_token_lvl": torch.ones(I, dtype=torch.bool),  # [I]
        }

        # ---- Assemble full example ----
        example = {
            "example_id": f"synthetic_{idx}",
            "feats": feats,
            "t": t,
            "noise": noise,
            "coord_atom_lvl_to_be_noised": coord_atom_lvl_to_be_noised,
            "ground_truth": ground_truth,
            "automorphisms": {},
            "symmetry_resolution": {},
            "extra_info": {},
        }

        return example
