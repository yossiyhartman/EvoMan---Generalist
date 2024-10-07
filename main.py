from trainer import Trainer

enemy_sets = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8]
]

# Loop over each set of enemies and train
for i, enemies in enumerate(enemy_sets):
    print(f"\nTraining on enemy set {i + 1}: {enemies}\n")

    trainer = Trainer(enemies)
    trainer.run()
    
    print(f"\nFinished training on enemy set {i + 1}: {enemies}\n")
