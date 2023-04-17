# LBA_Growing_Knowledge
Knowledge Learning by Asking


## Install

```bash
git clone https://github.com/greeksharifa/LBA_Growing_Knowledge.git

git submodule foreach init
git submodule foreach update

# git submodule foreach 'git pull'
```

**참고: git submodule** 
- https://data-engineer-tech.tistory.com/20
- https://github.com/madeleinegrunde/AGQA_baselines_code
- https://github.com/JingweiJ/ActionGenome.git

## Data Preparation

- AGQA: https://cs.stanford.edu/people/ranjaykrishna/agqa/
- Action Genome: https://www.actiongenome.org/#download
- Charades: https://prior.allenai.org/projects/charades

### Action Genome
data에 대한 symbolic link 생성
```bash
ln -s <original path> <link path>
ln -s ~/data/Charades/Charades_v1_480/  dataset/ag/videos
ln -s ~/data/Action_Genome/annotations/ dataset/ag/annotations
```
