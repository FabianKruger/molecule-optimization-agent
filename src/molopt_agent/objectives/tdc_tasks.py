from typing import Literal

from ..oracles.base import Oracle, OracleResult
from ..state import WorkflowState
from .generic_messages import get_generic_first_message


# Common first message template structure
FIRST_MESSAGE_TEMPLATE = """{task_description}

Objective:
- {direction} the {metric_name}.
- You will have {max_iterations} iterations to optimize the molecule.

Step 1:
{step_instruction}

Respond with a single JSON object:
{{
  "reason": "<why this is a reasonable starting point for this objective>",
  "smiles": "<SMILES string>"
}}"""


# Dictionary mapping TDC oracle names to their specific descriptions
TDC_ORACLE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "celecoxib_rediscovery": {
        "task_description": (
            "Rediscovery task: Rediscover a hidden molecule by maximizing similarity to the target molecule (score in [0,1], higher is better)."
        ),
        "metric_name": "similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will achieve a high similarity score to the hidden reference molecule."
        ),
    },
    "troglitazone_rediscovery": {
        "task_description": (
            "Rediscovery task: Rediscover a hidden molecule by maximizing similarity to the target molecule (score in [0,1], higher is better)."
        ),
        "metric_name": "similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will achieve a high similarity score to the hidden reference molecule."
        ),
    },
    "thiothixene_rediscovery": {
        "task_description": (
            "Rediscovery task: Rediscover a hidden molecule by maximizing similarity to the target molecule (score in [0,1], higher is better)."
        ),
        "metric_name": "similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will achieve a high similarity score to the hidden reference molecule."
        ),
    },

    "albuterol_similarity": {
        "task_description": (
            "Similarity task: Maximize similarity to a hidden target molecule (score in [0,1], higher is better).\n\n"
            "Scoring: score = min(1.0, similarity / 0.75)\n"
            "This means similarity ≥ 0.75 gives the maximum score of 1.0.\n\n"
        ),
        "metric_name": "thresholded similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have high thresholded similarity to the hidden target molecule."
        ),
    },
    "mestranol_similarity": {
        "task_description": (
            "Similarity task: Maximize similarity to a hidden target molecule (score in [0,1], higher is better).\n\n"
            "Scoring: score = min(1.0, similarity / 0.75)\n"
            "This means similarity ≥ 0.75 gives the maximum score of 1.0.\n\n"
        ),
        "metric_name": "thresholded similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have high thresholded similarity to the hidden target molecule."
        ),
    },

    # Isomers (explicit formula targets)
    "isomers_c7h8n2o2": {
        "task_description": (
            "Isomer-generation task: generate molecules matching the molecular formula C7H8N2O2. "
            "Perfect score corresponds to exactly matching the target element counts (and total atom count). "
        ),
        "metric_name": "formula-match score for C7H8N2O2 (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think matches the target molecular formula C7H8N2O2 exactly."
        ),
    },
    "isomers_c9h10n2o2pf2cl": {
        "task_description": (
            "Isomer-generation task: generate molecules matching the molecular formula C9H10N2O2PF2Cl. "
            "Perfect score corresponds to exactly matching the target element counts (and total atom count). "
        ),
        "metric_name": "formula-match score for C9H10N2O2PF2Cl (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think matches the target molecular formula C9H10N2O2PF2Cl exactly."
        ),
    },

    # Median molecules (maximize geometric mean of two similarities; targets provided)
    "median1": {
        "task_description": (
            "Median molecules: maximize the geometric mean of similarities to Camphor and Menthol.\n"
            "Target SMILES:\n"
            "- Camphor: CC1(C)C2CCC1(C)C(=O)C2 \n"
            "- Menthol: CC(C)C1CCC(C)CC1O \n\n"
            "Scoring: score = geom_mean(sim(camphor, ECFC4), sim(menthol, ECFC4))."
        ),
        "metric_name": "geometric-mean similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will be simultaneously similar to both (high similarity to Camphor and to Menthol)."
        ),
    },
    "median2": {
        "task_description": (
            "Median molecules: maximize the geometric mean of similarities to Tadalafil and Sildenafil.\n"
            "Target SMILES:\n"
            "- Tadalafil: O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C \n"
            "- Sildenafil: CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C \n\n"
            "Scoring: score = geom_mean(sim(tadalafil, ECFC6), sim(sildenafil, ECFC6))."
        ),
        "metric_name": "geometric-mean similarity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will be simultaneously similar to both (high similarity to Tadalafil and to Sildenafil)."
        ),
    },

    # MPO tasks (explicit composite components; geometric mean; Gaussian/Thresholded modifiers)
    "osimertinib_mpo": {
        "task_description": (
            "Osimertinib MPO: score is the geometric mean of 4 terms (each in [0,1]):\n"
            "  score = gmean([sim_v1, sim_v2, tpsa_term, logp_term])\n\n"
            "Reference molecule (Osimertinib) SMILES:\n"
            "  COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34 \n\n"
            "Terms:\n"
            "  sim_v1  = clip(Tanimoto(FCFP4(test), FCFP4(ref)), t=0.8)\n"
            "  sim_v2  = min_gauss(Tanimoto(ECFP6(test), ECFP6(ref)); mu=0.85, sigma=0.1)\n"
            "  tpsa_term = max_gauss(TPSA(test); mu=100, sigma=10)\n"
            "  logp_term = min_gauss(MolLogP(test); mu=1, sigma=1)\n\n"
            "Modifiers:\n"
            "  clip(x,t) = 1 if x>=t else x/t\n"
            "  min_gauss(x;mu,s) = 1 if x<=mu else exp(-0.5*((x-mu)/s)^2)\n"
            "  max_gauss(x;mu,s) = 1 if x>=mu else exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Osimertinib MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Osimertinib under BOTH similarity terms (FCFP4 similarity clipped at 0.8, and ECFP6 similarity favored up to ~0.85),\n"
            "- having TPSA in the high-preferred region (max-Gaussian with target around 100), and\n"
            "- having logP in the low-preferred region (min-Gaussian with target around 1)."
        )
    },

    "fexofenadine_mpo": {
        "task_description": (
            "Fexofenadine MPO: score is the geometric mean of 3 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, tpsa_term, logp_term])\n\n"
            "Reference molecule (Fexofenadine) SMILES:\n"
            "  CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4 \n\n"
            "Terms:\n"
            "  similarity_term = clip(Tanimoto(AP(test), AP(ref)), t=0.8)\n"
            "  tpsa_term = max_gauss(TPSA(test); mu=90, sigma=10)\n"
            "  logp_term = min_gauss(MolLogP(test); mu=4, sigma=1)\n\n"
            "Modifiers:\n"
            "  clip(x,t) = 1 if x>=t else x/t\n"
            "  min_gauss(x;mu,s) = 1 if x<=mu else exp(-0.5*((x-mu)/s)^2)\n"
            "  max_gauss(x;mu,s) = 1 if x>=mu else exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Fexofenadine MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Fexofenadine in AP fingerprint similarity (clipped at 0.8),\n"
            "- having TPSA in the high-preferred region (max-Gaussian with target around 90), and\n"
            "- having logP in the low-preferred region (min-Gaussian with target around 4)."
        )
    },

    "ranolazine_mpo": {
        "task_description": (
            "Ranolazine MPO: score is the geometric mean of 4 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, tpsa_term, logp_term, fluorine_term])\n\n"
            "Reference molecule (Ranolazine) SMILES:\n"
            "  COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2 \n\n"
            "Terms:\n"
            "  similarity_term = clip(Tanimoto(AP(test), AP(ref)), t=0.7)\n"
            "  tpsa_term = max_gauss(TPSA(test); mu=95, sigma=20)\n"
            "  logp_term = max_gauss(MolLogP(test); mu=7, sigma=1)\n"
            "  fluorine_term = gauss(num_F_atoms(test); mu=1, sigma=1.0)\n\n"
            "Modifiers:\n"
            "  clip(x,t) = 1 if x>=t else x/t\n"
            "  gauss(x;mu,s) = exp(-0.5*((x-mu)/s)^2)\n"
            "  max_gauss(x;mu,s) = 1 if x>=mu else exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Ranolazine MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Ranolazine in AP fingerprint similarity (clipped at 0.7),\n"
            "- having TPSA in the high-preferred region (max-Gaussian with target around 95),\n"
            "- having logP in the high-preferred region (max-Gaussian with target around 7), and\n"
            "- having about 1 fluorine atom (Gaussian with target 1)."
        )
    },

    "perindopril_mpo": {
        "task_description": (
            "Perindopril MPO: score is the geometric mean of 2 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, aromatic_rings_term])\n\n"
            "Reference molecule (Perindopril) SMILES:\n"
            "  O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC \n\n"
            "Terms:\n"
            "  similarity_term = Tanimoto(ECFP4(test), ECFP4(ref))  (no clipping/modifier)\n"
            "  aromatic_rings_term = gauss(CalcNumAromaticRings(test); mu=2, sigma=0.5)\n\n"
            "Modifier:\n"
            "  gauss(x;mu,s) = exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Perindopril MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Perindopril in ECFP4 Tanimoto similarity, and\n"
            "- having about 2 aromatic rings (Gaussian with target 2)."
        )
    },

    "amlodipine_mpo": {
        "task_description": (
            "Amlodipine MPO: score is the geometric mean of 2 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, rings_term])\n\n"
            "Reference molecule (Amlodipine) SMILES:\n"
            "  Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\\C(=O)OC \n\n"
            "Terms:\n"
            "  similarity_term = Tanimoto(ECFP4(test), ECFP4(ref))  (no clipping/modifier)\n"
            "  rings_term = gauss(CalcNumRings(test); mu=3, sigma=0.5)\n\n"
            "Modifier:\n"
            "  gauss(x;mu,s) = exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Amlodipine MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Amlodipine in ECFP4 Tanimoto similarity, and\n"
            "- having about 3 rings total (Gaussian with target 3)."
        )
    },

    "sitagliptin_mpo": {
        "task_description": (
            "Sitagliptin MPO: score is the geometric mean of 4 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, logp_term, tpsa_term, isomer_term])\n\n"
            "Reference molecule (Sitagliptin) SMILES (used to define fingerprint + target logP/TPSA):\n"
            "  Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F \n\n"
            "Terms:\n"
            "  similarity_term = gauss(sim; mu=0, sigma=0.1)\n"
            "    where sim = Tanimoto(ECFP4(test), ECFP4(ref))\n"
            "    (this rewards LOW similarity: best when sim is near 0)\n"
            "  logp_term = gauss(MolLogP(test); mu=MolLogP(ref), sigma=0.2)\n"
            "  tpsa_term = gauss(TPSA(test); mu=TPSA(ref), sigma=5)\n"
            "  isomer_term = Isomer_scoring('C16H15F6N5O')(test_smiles)\n\n"
            "Modifier:\n"
            "  gauss(x;mu,s) = exp(-0.5*((x-mu)/s)^2)"
        ),
        "metric_name": "Sitagliptin MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- matching the target molecular formula C16H15F6N5O (isomer scoring term),\n"
            "- having logP close to Sitagliptin’s logP (Gaussian around the reference logP),\n"
            "- having TPSA close to Sitagliptin’s TPSA (Gaussian around the reference TPSA), and\n"
            "- being DISSIMILAR to Sitagliptin in ECFP4 Tanimoto similarity (Gaussian centered at similarity 0)."
        )
    },

    "zaleplon_mpo": {
        "task_description": (
            "Zaleplon MPO: score is the geometric mean of 2 terms (each in [0,1]):\n"
            "  score = gmean([similarity_term, isomer_term])\n\n"
            "Reference molecule (Zaleplon) SMILES:\n"
            "  O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1 \n\n"
            "Terms:\n"
            "  similarity_term = Tanimoto(ECFP4(test), ECFP4(ref))\n"
            "  isomer_term = Isomer_scoring('C19H17N3O2')(test_smiles)"
        ),
        "metric_name": "Zaleplon MPO score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high MPO score by:\n"
            "- being similar to Zaleplon in ECFP4 Tanimoto similarity, and\n"
            "- matching the target molecular formula C19H17N3O2 (isomer scoring term)."
        )
    },

    # SMARTS-constrained objective
    "valsartan_smarts": {
        "task_description": (
            "Valsartan SMARTS task: Design a molecule containing a required substructure while matching property targets.\n\n"
            "Critical constraint: Required substructure (SMARTS pattern):\n"
            "  CN(C=O)Cc1ccc(c2ccccc2)cc1\n"
            "Your molecule MUST contain this exact substructure as a subgraph match.\n\n"
            "SCORING: The score is the geometric mean of 4 terms:\n"
            "  score = (smarts_term * logp_term * tpsa_term * bertz_term)^(1/4)\n\n"
            "Because this is a geometric mean, if smarts_term = 0, the entire score = 0,\n"
            "no matter how good the other properties are!\n\n"
            "Term definitions:\n"
            "  smarts_term = 1.0 if the molecule contains the SMARTS pattern, else 0.0\n"
            "  logp_term   = exp(-0.5 * ((MolLogP(mol) - target_logp) / 0.2)^2)\n"
            "  tpsa_term   = exp(-0.5 * ((TPSA(mol) - target_tpsa) / 5)^2)\n"
            "  bertz_term  = exp(-0.5 * ((BertzCT(mol) - target_bertz) / 30)^2)\n\n"
            "The target property values come from Sitagliptin (NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F). \n\n"
            "BertzCT measures molecular complexity (larger, more complex molecules have higher values)."
        ),
        "metric_name": "Valsartan SMARTS composite score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high score by:\n"
            "- containing the required SMARTS pattern (otherwise the score is 0), and\n"
            "- having MolLogP, TPSA, and BertzCT close to the reference molecule’s values (as judged by the Gaussian terms)."
        ),
    },


    # Hop tasks (arithmetic mean; SMARTS include/exclude + similarity-to-reference with threshold)
    "deco_hop": {
        "task_description": (
            "Decorator Hop task: score is the arithmetic mean of 4 terms:\n"
            "  score = mean([similarity_term, deco1_term, deco2_term, scaffold_term])\n\n"
            "Reference molecule used for similarity (pharmacophore) SMILES:\n"
            "  CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C\n\n"
            "Similarity term:\n"
            "  - Based on PHCO (2D pharmacophore fingerprint).\n"
            "  - sim = Tanimoto(PHCO(test), PHCO(reference))\n"
            "  - similarity_term = clip(sim, t=0.85) where clip(x,t)=1 if x>=t else x/t\n\n"
            "SMARTS constraint terms (each is binary 0/1):\n"
            "  - deco1_term = 1 if SMARTS 'CS([#6])(=O)=O' is ABSENT, else 0\n"
            "  - deco2_term = 1 if SMARTS '[#7]-c1ccc2ncsc2c1' is ABSENT, else 0\n"
            "  - scaffold_term = 1 if SMARTS '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12' is PRESENT, else 0"
        ),
        "metric_name": "Deco Hop score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high score by:\n"
            "- achieving high PHCO pharmacophore similarity to the reference (ideally sim >= 0.85 so the clipped term is 1),\n"
            "- NOT containing 'CS([#6])(=O)=O',\n"
            "- NOT containing '[#7]-c1ccc2ncsc2c1', and\n"
            "- DO containing '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12'."
        ),
    },
    
    "scaffold_hop": {
        "task_description": (
            "Scaffold Hop task: score is the arithmetic mean of 3 terms:\n"
            "  score = mean([similarity_term, deco_term, scaffold_term])\n\n"
            "Reference molecule used for similarity (pharmacophore) SMILES:\n"
            "  CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C\n\n"
            "Similarity term:\n"
            "  - Based on PHCO (2D pharmacophore fingerprint).\n"
            "  - sim = Tanimoto(PHCO(test), PHCO(reference))\n"
            "  - similarity_term = clip(sim, t=0.75) where clip(x,t)=1 if x>=t else x/t\n\n"
            "SMARTS constraint terms (each is binary 0/1):\n"
            "  - deco_term = 1 if SMARTS '[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1' is PRESENT, else 0\n"
            "  - scaffold_term = 1 if SMARTS '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12' is ABSENT, else 0"
        ),
        "metric_name": "Scaffold Hop score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think will have a high score by:\n"
            "- achieving high PHCO pharmacophore similarity to the reference (ideally sim >= 0.75 so the clipped term is 1),\n"
            "- DO containing '[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1', and\n"
            "- NOT containing '[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12'."
        ),
    },
        "qed": {
        "task_description": (
            "Optimize RDKit QED (Quantitative Estimate of Drug-likeness), a normalized score in [0, 1] "
            "where higher is better and 1 is optimal. QED evaluates 8 molecular properties:\n"
            "1. Molecular Weight (optimal: 281-332 Da)\n"
            "2. LogP/lipophilicity (optimal: 1.95-3.57)\n"
            "3. Hydrogen Bond Acceptors (optimal: 2-3)\n"
            "4. Hydrogen Bond Donors (optimal: 1)\n"
            "5. Polar Surface Area (optimal: 40-69 Ų)\n"
            "6. Rotatable Bonds (optimal: 2-4)\n"
            "7. Aromatic Rings (optimal: 1-2)\n"
            "8. Structural Alerts (optimal: 0 - no problematic substructures)"
        ),
        "metric_name": "QED score (0–1; higher is better, 1 is optimal)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that has a high QED score. "
        ),
    },
    "drd2": {
        "task_description": (
            "Optimize predicted bioactivity against Dopamine Receptor D2 (DRD2). "
            "Scores are in [0, 1] with higher being better."
        ),
        "metric_name": "predicted DRD2 activity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think has good predicted bioactivity against DRD2."
        ),
    },
    "gsk3b": {
        "task_description": (
            "Optimize predicted bioactivity against Glycogen Synthase Kinase 3 beta (GSK3β). "
            "Scores are in [0, 1] with higher being better."
        ),
        "metric_name": "predicted GSK3β activity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think has good predicted bioactivity against GSK3β."
        ),
    },
    "jnk3": {
        "task_description": (
            "Optimize predicted bioactivity against c-Jun N-terminal kinase 3 (JNK3), which belongs to the mitogen-activated protein kinase family. "
            "Scores are in [0, 1] with higher being better."
        ),
        "metric_name": "predicted JNK3 activity score (0–1; higher is better)",
        "step_instruction": (
            "Propose a single initial molecule as a SMILES string that you think has good predicted bioactivity against JNK3."
        ),
    },
}



# Default descriptions for unknown TDC oracles
DEFAULT_ORACLE_DESCRIPTIONS = {
    "task_description": "We are optimizing molecules using a TDC oracle.",
    "metric_name": "predicted score",
    "step_instruction": "Propose a single initial molecule as a SMILES string that you expect to perform well on this objective.",
}


class TdcObjective:
    """
    Objective for optimizing molecules using any TDC oracle.
    
    Supports both minimization and maximization objectives.
    Always runs for the full number of max_iterations.
    """

    name = "tdc_tasks"

    def __init__(
        self,
        oracle: Oracle,
        direction: Literal["minimize", "maximize"] = "maximize",
        max_iterations: int = 20,
        first_message_template: str | None = None,
        xai: Literal["none", "no_description"] = "none",
    ):
        """
        Initialize TDC objective.
        
        Args:
            oracle: The TDC oracle to use
            direction: Whether to minimize or maximize the score
            max_iterations: Maximum number of iterations (always runs full number)
            first_message_template: Optional custom first message template.
                                   If None, uses template from TDC_ORACLE_DESCRIPTIONS
                                   or DEFAULT_ORACLE_DESCRIPTIONS
            xai: XAI mode. "none" shows task description, "no_description" hides it.
        """
        self.oracle = oracle
        self._direction = direction
        self._max_iterations = int(max_iterations)
        self._first_message_template = first_message_template
        self._xai = xai

        # Extract oracle name from oracle if possible
        oracle_name = None
        if hasattr(oracle, "get_params"):
            params = oracle.get_params()
            oracle_name = params.get("oracle_name")

        # Get oracle-specific descriptions
        if self._first_message_template is None:
            if oracle_name and oracle_name in TDC_ORACLE_DESCRIPTIONS:
                self._oracle_descriptions = TDC_ORACLE_DESCRIPTIONS[oracle_name]
            else:
                self._oracle_descriptions = DEFAULT_ORACLE_DESCRIPTIONS
        else:
            # If custom template provided, store it for direct use
            self._oracle_descriptions = None

    def first_message(self) -> str:
        """Generate the first message for the optimization task."""
        higher_is_better = self._direction == "maximize"
        
        # Use generic message for no_description mode
        if self._xai == "no_description":
            return get_generic_first_message(higher_is_better=higher_is_better)
        
        direction_text = "maximize" if self._direction == "maximize" else "minimize"
        
        # If custom template was provided, use it directly
        if self._first_message_template is not None:
            return self._first_message_template.format(
                direction=direction_text,
                max_iterations=self._max_iterations,
            )
        
        # Otherwise, build from modular components
        return FIRST_MESSAGE_TEMPLATE.format(
            task_description=self._oracle_descriptions["task_description"],
            direction=direction_text,
            metric_name=self._oracle_descriptions["metric_name"],
            max_iterations=self._max_iterations,
            step_instruction=self._oracle_descriptions["step_instruction"],
        )

    def evaluate(self, state: WorkflowState) -> OracleResult:
        """Evaluate the current molecule using the oracle."""
        smiles = state["current_smiles"]
        return self.oracle(smiles)

    def build_feedback(self, state: WorkflowState, result: OracleResult) -> str:
        """Build feedback message for the agent."""
        score = result["score"]
        direction_text = "maximize" if self._direction == "maximize" else "minimize"

        return f"""
SMILES: {state['current_smiles']}
Score: {score:.4f} (direction: {direction_text})
Iteration: {state['iteration_count']} / {self._max_iterations}

Respond with JSON only:
{{
  "reason": "<what you changed and why>",
  "smiles": "<new SMILES>"
}}
""".strip()

    def is_done(self, state: WorkflowState, result: OracleResult) -> bool:
        """Check if optimization is complete."""
        # Always run for the full number of iterations
        return state["iteration_count"] >= self._max_iterations

    def max_iterations(self) -> int:
        """Return maximum number of iterations."""
        return self._max_iterations
