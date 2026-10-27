from dataclasses import dataclass


from typing import Dict, Tuple





import torch


from torch import Tensor





from chaw_op.contracts import FieldBattery








@dataclass(frozen=True)


class BiomarkerComponents:


    low_shear_burden: Tensor


    oscillatory_burden: Tensor


    ecap_burden: Tensor


    ilt_burden: Tensor


    peak_wall_stress: Tensor


    rupture_potential: Tensor


    coupled_overlap: Tensor


    integrated: Tensor








def segment_reduce(values: Tensor, batch: Tensor, operation: str) -> Tensor:


    groups = int(batch.max().item()) + 1


    results = []


    for group in range(groups):


        selected = values[batch == group]


        if operation == "mean":


            results.append(selected.mean(dim=0))


        elif operation == "max":


            results.append(selected.max(dim=0).values)


        elif operation == "q95":


            results.append(torch.quantile(selected, 0.95, dim=0))


        else:


            raise ValueError(operation)


    return torch.stack(results)








def integrate_biomarker(fields: FieldBattery, batch: Tensor) -> BiomarkerComponents:


    tawss_reference = segment_reduce(fields.tawss, batch, "mean")[batch].clamp_min(1e-6)


    pws_reference = segment_reduce(fields.pws, batch, "mean")[batch].clamp_min(1e-6)


    low_shear = torch.sigmoid((tawss_reference - fields.tawss) / tawss_reference)


    oscillatory = fields.osi / 0.5


    ecap_scaled = fields.ecap / segment_reduce(fields.ecap, batch, "q95")[batch].clamp_min(1e-6)


    pws_scaled = fields.pws / pws_reference


    overlap_field = low_shear * fields.ilt * pws_scaled


    low_shear_burden = segment_reduce(low_shear, batch, "mean")


    oscillatory_burden = segment_reduce(oscillatory, batch, "mean")


    ecap_burden = segment_reduce(ecap_scaled, batch, "mean")


    ilt_burden = segment_reduce(fields.ilt, batch, "mean")


    peak_wall_stress = segment_reduce(fields.pws, batch, "q95")


    rupture_potential = segment_reduce(fields.rpi, batch, "q95")


    coupled_overlap = segment_reduce(overlap_field, batch, "mean")


    integrated = (


        0.15 * low_shear_burden


        + 0.10 * oscillatory_burden


        + 0.15 * ecap_burden


        + 0.15 * ilt_burden


        + 0.15 * peak_wall_stress


        + 0.10 * rupture_potential


        + 0.20 * coupled_overlap


    )


    return BiomarkerComponents(


        low_shear_burden,


        oscillatory_burden,


        ecap_burden,


        ilt_burden,


        peak_wall_stress,


        rupture_potential,


        coupled_overlap,


        integrated,


    )








def component_matrix(components: BiomarkerComponents) -> Tensor:


    return torch.cat(


        [


            components.low_shear_burden,


            components.oscillatory_burden,


            components.ecap_burden,


            components.ilt_burden,


            components.peak_wall_stress,


            components.rupture_potential,


            components.coupled_overlap,


        ],


        dim=-1,


    )


