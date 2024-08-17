from tmtools.io import get_structure, get_residue_data
from tmtools.testing import get_pdb_path
from tmtools import tm_align
from tmtools.io import get_residue_data
from Bio.Data import IUPACData
import numpy as np
# 扩展 protein_letters_3to1 字典
extended_protein_letters_3to1 = IUPACData.protein_letters_3to1_extended.copy()
extended_protein_letters_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'SEP': 'S',  # 磷酸化丝氨酸
    'TPO': 'T',  # 磷酸化苏氨酸
    'PYL': 'O',  # 吡咯赖氨酸
    'SEC': 'U',  # 硒代半胱氨酸
    # 添加更多非标准氨基酸映射
}
def TMalign(s1,s2):
    s1 = get_structure(get_pdb_path(s1))
    s2 = get_structure(get_pdb_path(s2))
    chain1 = next(s1.get_chains())
    chain2 = next(s2.get_chains())
    # 用于支持非标准残基的获取残基数据的函数
    def get_residue_data_extended(chain):
        coords = []
        seq = []
        for residue in chain.get_residues():
            if "CA" in residue.child_dict:
                coords.append(residue.child_dict["CA"].coord)
                # 使用扩展字典进行转换
                seq.append(extended_protein_letters_3to1.get(residue.resname, 'X'))  # 对未知残基使用 'X'
        return np.vstack(coords), "".join(seq)

    # 现在，使用这个函数代替原始的 get_residue_data
    coords1, seq1 = get_residue_data_extended(chain1)
    coords2, seq2 = get_residue_data_extended(chain2)
    res = tm_align(coords1, coords2, seq1, seq2)
    return res.tm_norm_chain1,res.tm_norm_chain2