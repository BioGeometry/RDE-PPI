import os
import copy
import argparse
import pandas as pd
from pathlib import Path
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio.SeqUtils import seq1
from Bio.PDB import Structure
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm

from rde.utils.misc import load_config, seed_all
from rde.utils.data import PaddingCollate
from rde.utils.train import *
from rde.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch
from rde.utils.protein.parsers import parse_biopython_structure
from rde.models.rde_ddg import DDG_RDE_Network



_RESI_TYPE1_LIST = list("ARNDCQEGHILKMFPSTWYV")


class ResPdbId:
    def __init__(self, resseq: int, icode: str = ' '):
        self.resseq = resseq
        self.icode = icode

    @classmethod
    def from_str(cls, res_str: str):
        res_str = res_str.rstrip()
        last_char = res_str[-1]
        if last_char.isalpha():
            resseq = int(res_str[:-1])
            icode = last_char
            return cls(resseq, icode)
        else:
            return cls(int(res_str))

    def __lt__(self, other):
        return self.resseq < other.resseq or (self.resseq == other.resseq and self.icode < other.icode)

    def __eq__(self, other):
        return self.resseq == other.resseq and self.icode == other.icode

    def __le__(self, other):
        return self < other or self == other

    def __repr__(self):
        return f'{self.resseq}{self.icode}'.strip()

    def __hash__(self) -> int:
        return hash(self.resseq) + hash(self.icode)


def get_all_single_mutations(pdb_struct: Structure, seg_chain: str, res_id_start: str, res_id_end: str):
    """
    Generate all possible single mutations for a given chain segment.

    Args:
        pdb_struct: biopython Structure representing the protein structure.
        seg_chain: id of the chain to be mutated.
        res_id_start: starting residue id of the chain segment.
        res_id_end: ending residue id of the chain segment.

    Returns:
        mutations: list of all possible single mutations for the given chain segment.
        positions: list of all possible mutation positions for the given chain segment.
    """
    mutations, positions = [], []
    for res in pdb_struct[0][seg_chain]:
        het, resseq, icode = res.get_id()
        res_pdbid = ResPdbId(resseq, icode)
        if ResPdbId.from_str(res_id_start) <= res_pdbid <= ResPdbId.from_str(res_id_end):
            wt_type = seq1(res.get_resname())
            if wt_type in _RESI_TYPE1_LIST:
                # residue in segment and has a foldx-valid type, mutate it
                positions.append(f'{wt_type}{seg_chain}{res_pdbid}*')
                for mt_type in _RESI_TYPE1_LIST:
                    if wt_type == mt_type:
                        continue
                    mutations.append(f'{wt_type}{seg_chain}{res_pdbid}{mt_type}')
    return mutations, positions


def get_structure(pdb_path):
    pdb_path = Path(pdb_path).expanduser()
    if pdb_path.suffix == '.pdb':
        parser = PDBParser(QUIET=True)
    elif pdb_path.suffix == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError('Unknown file type.')

    structure = parser.get_structure(None, pdb_path)
    return structure


class PMDataset(Dataset):

    def __init__(self, pdb_path, mutations):
        super().__init__()
        self.pdb_path = pdb_path

        self.data = None
        self.seq_map = None
        self._load_structure()

        self.mutations = self._parse_mutations(mutations)
        self.transform = Compose([
            SelectAtom('backbone+CB'),
            SelectedRegionFixedSizePatch('mut_flag', 128)
        ])


    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_structure(self):
        structure = get_structure(self.pdb_path)
        data, seq_map = parse_biopython_structure(structure[0])
        self.data = data
        self.seq_map = seq_map

    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])
            pos = (ch, seq, ' ')
            if pos not in self.seq_map: continue

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'position': pos,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'position': pos,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        # Below is the obsolete code for single mutation
        data = self.clone_data()
        mut = self.mutations[index]
        mut_pos_idx = self.seq_map[mut['position']]

        data['mut_flag'] = torch.zeros(size=data['aa'].shape, dtype=torch.bool)
        data['mut_flag'][mut_pos_idx] = True
        data['aa_mut'] = data['aa'].clone()
        data['aa_mut'][mut_pos_idx] = one_to_index(mut['mt'])
        data = self.transform(data)
        data['ddG'] = 0
        data['mutstr'] = '{}{}{}{}'.format(
            mut['wt'],
            mut['position'][0],
            mut['position'][1],
            mut['mt']
        )
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--pdb', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--mutations', type=str, help='List of mutations in the format of "{wt_aa}{chain_id}{res_id}{mt_aa}", e.g. "IH93*,AL63W"')
    group.add_argument('-s', '--segments', type=str, help='List of segments to perform saturation mutagenesis, in the format of "{chain_id}:{start_res_id}-{end_res_id}", e.g. "A:10-20,B:15-15"')
    parser.add_argument('-c', '--checkpoint', default=os.path.join(os.path.dirname(__file__), 'trained_models/DDG_RDE_Network_30k.pt'))
    parser.add_argument('--interest', type=str, nargs='+', help='List of mutations to show results for.')
    parser.add_argument('-o', '--output', type=str, default='pm_results.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Model
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=ckpt['config'], num_cvfolds=3)
    cv_mgr.load_state_dict(ckpt['model'])
    cv_mgr.to(args.device)
    model = None

    # Data
    if args.segments:
        mutations = []
        for seg in args.segments.split(','):
            chain_id, res_id_range = seg.split(':')
            res_id_start, res_id_end = res_id_range.split('-')
            mutations.extend(get_all_single_mutations(
                get_structure(args.pdb),
                chain_id, res_id_start, res_id_end
            )[1])
    else:
        mutations = args.mutations.split(',')
    print(mutations)
    dataset = PMDataset(
        pdb_path = args.pdb,
        mutations = mutations
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PaddingCollate(),
    )

    result = []
    for batch in tqdm(loader):
        batch = recursive_to(batch, args.device)
        for fold in range(cv_mgr.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            with torch.no_grad():
                _, out_dict = model(batch)
            for mutstr, ddG_pred in zip(batch['mutstr'], out_dict['ddG_pred'].cpu().tolist()):
                result.append({
                    'mutstr': mutstr,
                    'ddG_pred': ddG_pred,
                })
    result = pd.DataFrame(result)
    result = result.groupby('mutstr').mean().reset_index()
    result['rank'] = result['ddG_pred'].rank() / len(result)
    print(result)
    print(f'Results saved to {args.output}.')
    result.to_csv(args.output)

    if args.interest:
        print(result[result['mutstr'].isin(args.interest)])
