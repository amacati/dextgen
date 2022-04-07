# Master Thesis Martin Schuck

[![PEP8 Check](https://github.com/amacati/rl/actions/workflows/linting.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/linting.yaml)
[![Tests](https://github.com/amacati/rl/actions/workflows/testing.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/testing.yaml)

## Clear saves folder from backups

Run the following command from the RL root folder.

> :warning: **WARNING**: Make sure you execute from the correct folder or this may have catastrophic consequences!

```$ find . -type d -name backup -exec rm -r "{}" \;```

## Planned environments

1. HER
   - FlatPJCube
    
2. Pretraining
   - Flat3FSphere
   - Flat3FCube
   - Flat3FCylinder
   - Flat3FMesh
    
3. Eigengrasps
   - Flat3FCube
   - Flat3FCylinder
   - Flat3FMesh
    
   - FlatSHCube
   - FlatSHCylinder
   - FlatSHMesh
    
4. Total learning
   - pretraining
     - FlatPJSphere
     - FlatPJCube
     - FlatPJCylinder
     - FlatPJMesh
      
     - Flat3FSphere
     - Flat3FCube
     - Flat3FCylinder
     - Flat3FMesh
      
     - FlatSHSphere
     - FlatSHCube
     - FlatSHCylinder
     - FlatSHMesh
    
   - Sphere no pretraining
     - FlatPJSphere
     - Flat3FSphere
     - FlatSHSphere
    
5. Complex environments
   - Obstacle region
     - FlatObstacleSHCube
    
   - Uneven ground
     - Uneven3FCube
     - Uneven3FMesh
     - UnevenSHCube
     - UnevenSHMesh
    
6. SeaClear
   - UnevenSCMesh