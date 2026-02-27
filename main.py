import math
import random
import statistics
from dataclasses import dataclass

import pygame

# =========================
# Config
# =========================
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
UI_HEIGHT = 100
WORLD_WIDTH = 60
WORLD_HEIGHT = 40
TILE_SIZE = min(SCREEN_WIDTH // WORLD_WIDTH, (SCREEN_HEIGHT - UI_HEIGHT) // WORLD_HEIGHT)

POP_SIZE = 80
ELITE_COUNT = 8
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.08
MUTATION_STD = 0.25
BIG_MUTATION_CHANCE = 0.015
BIG_MUTATION_STD = 0.9
CROSSOVER_RATE = 0.9

STEPS_PER_EPISODE = 900
STEPS_SPEEDS = [1, 10, 50]

TREE_DENSITY = 0.15
FOOD_COUNT = 90
WATER_COUNT = 70
FOOD_RESPAWN_CHANCE = 0.015
WATER_RESPAWN_CHANCE = 0.012

BASE_ENERGY = 65.0
MAX_ENERGY = 120.0
MOVE_COST = 0.35
IDLE_COST = 0.23
HIT_WALL_PENALTY = 0.08
FOOD_ENERGY_GAIN = 22.0
WATER_ENERGY_GAIN = 18.0

FIT_SURVIVAL = 0.06
FIT_FOOD = 7.0
FIT_WATER = 6.5
FIT_EXPLORE = 0.08
FIT_IDLE_PENALTY = 0.05
FIT_STUCK_PENALTY = 0.03

INPUT_SIZE = 19
HIDDEN_SIZE = 18
OUTPUT_SIZE = 5  # up/down/left/right/stay

BG_COLOR = (20, 24, 20)
TREE_COLOR = (35, 80, 35)
FOOD_COLOR = (220, 180, 60)
WATER_COLOR = (70, 130, 230)
EMPTY_COLOR = (35, 40, 35)
AGENT_COLOR = (240, 80, 80)
UI_COLOR = (18, 18, 20)
GRID_COLOR = (30, 34, 30)
GRAPH_COLOR = (80, 220, 120)


@dataclass
class EpisodeResult:
    fitness: float
    lifespan: int
    food_eaten: int
    water_drank: int
    explored_cells: int
    final_energy: float


class NetworkGenome:
    def __init__(self, genes=None):
        self.gene_count = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
        if genes is None:
            self.genes = [random.uniform(-1.0, 1.0) for _ in range(self.gene_count)]
        else:
            self.genes = genes[:]

    def copy(self):
        return NetworkGenome(self.genes)

    def forward(self, inputs):
        idx = 0
        hidden = []
        for h in range(HIDDEN_SIZE):
            s = 0.0
            for i in range(INPUT_SIZE):
                s += inputs[i] * self.genes[idx]
                idx += 1
            s += self.genes[idx]
            idx += 1
            hidden.append(math.tanh(s))

        outputs = []
        for o in range(OUTPUT_SIZE):
            s = 0.0
            for h in range(HIDDEN_SIZE):
                s += hidden[h] * self.genes[idx]
                idx += 1
            s += self.genes[idx]
            idx += 1
            outputs.append(s)
        return outputs

    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < MUTATION_RATE:
                self.genes[i] += random.gauss(0.0, MUTATION_STD)
            if random.random() < BIG_MUTATION_CHANCE:
                self.genes[i] += random.gauss(0.0, BIG_MUTATION_STD)


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.trees = set()
        self.food = set()
        self.water = set()
        self.walkable = []
        self.reset()

    def reset(self):
        self.trees.clear()
        self.food.clear()
        self.water.clear()

        for y in range(self.height):
            for x in range(self.width):
                if random.random() < TREE_DENSITY:
                    self.trees.add((x, y))

        self.walkable = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.trees]
        if not self.walkable:
            self.trees.clear()
            self.walkable = [(x, y) for y in range(self.height) for x in range(self.width)]

        self._scatter_resources(self.food, FOOD_COUNT)
        self._scatter_resources(self.water, WATER_COUNT)

    def _scatter_resources(self, resource_set, count):
        tries = 0
        while len(resource_set) < count and tries < count * 20:
            tries += 1
            pos = random.choice(self.walkable)
            if pos in self.food or pos in self.water:
                continue
            resource_set.add(pos)

    def random_free_cell(self):
        tries = 0
        while tries < 200:
            tries += 1
            pos = random.choice(self.walkable)
            if pos not in self.food and pos not in self.water:
                return pos
        return random.choice(self.walkable)

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        return not self.in_bounds(x, y) or (x, y) in self.trees

    def nearest_vector(self, x, y, resource_set):
        if not resource_set:
            return 0.0, 0.0, 1.0
        best = None
        best_dist = float("inf")
        for rx, ry in resource_set:
            dx = rx - x
            dy = ry - y
            d = abs(dx) + abs(dy)
            if d < best_dist:
                best_dist = d
                best = (dx, dy)
        dx, dy = best
        maxd = self.width + self.height
        return dx / self.width, dy / self.height, min(1.0, best_dist / maxd)

    def maybe_respawn(self):
        if random.random() < FOOD_RESPAWN_CHANCE and len(self.food) < FOOD_COUNT:
            self._spawn_one(self.food)
        if random.random() < WATER_RESPAWN_CHANCE and len(self.water) < WATER_COUNT:
            self._spawn_one(self.water)

    def _spawn_one(self, target_set):
        for _ in range(100):
            pos = random.choice(self.walkable)
            if pos in self.food or pos in self.water:
                continue
            target_set.add(pos)
            break


class Agent:
    ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    def __init__(self, genome):
        self.genome = genome
        self.x = 0
        self.y = 0
        self.energy = BASE_ENERGY
        self.alive = True
        self.food_eaten = 0
        self.water_drank = 0
        self.explored = set()
        self.fitness = 0.0
        self.lifespan = 0
        self.idle_count = 0
        self.stuck_hits = 0

    def reset(self, world):
        self.x, self.y = world.random_free_cell()
        self.energy = BASE_ENERGY
        self.alive = True
        self.food_eaten = 0
        self.water_drank = 0
        self.explored = {(self.x, self.y)}
        self.fitness = 0.0
        self.lifespan = 0
        self.idle_count = 0
        self.stuck_hits = 0

    def observe(self, world):
        fx, fy, fd = world.nearest_vector(self.x, self.y, world.food)
        wx, wy, wd = world.nearest_vector(self.x, self.y, world.water)

        local = []
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                tx = self.x + ox
                ty = self.y + oy
                if not world.in_bounds(tx, ty) or (tx, ty) in world.trees:
                    local.append(-1.0)
                elif (tx, ty) in world.food:
                    local.append(0.5)
                elif (tx, ty) in world.water:
                    local.append(1.0)
                else:
                    local.append(0.0)

        energy_norm = self.energy / MAX_ENERGY
        noise = random.uniform(-1.0, 1.0)
        return [fx, fy, fd, wx, wy, wd] + local + [energy_norm, noise, 1.0, self.idle_count / 20.0]

    def step(self, world):
        if not self.alive:
            return

        inputs = self.observe(world)
        outputs = self.genome.forward(inputs)
        action = max(range(len(outputs)), key=lambda i: outputs[i])
        dx, dy = self.ACTIONS[action]

        nx, ny = self.x + dx, self.y + dy
        moved = False

        if action == 4:
            self.energy -= IDLE_COST
            self.idle_count += 1
            self.fitness -= FIT_IDLE_PENALTY
        elif world.is_blocked(nx, ny):
            self.energy -= (MOVE_COST + HIT_WALL_PENALTY)
            self.fitness -= FIT_STUCK_PENALTY
            self.stuck_hits += 1
            self.idle_count += 1
        else:
            self.x, self.y = nx, ny
            self.energy -= MOVE_COST
            moved = True
            self.idle_count = 0

        self.lifespan += 1
        self.fitness += FIT_SURVIVAL

        if moved and (self.x, self.y) not in self.explored:
            self.explored.add((self.x, self.y))
            self.fitness += FIT_EXPLORE

        pos = (self.x, self.y)
        if pos in world.food:
            world.food.remove(pos)
            self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY_GAIN)
            self.food_eaten += 1
            self.fitness += FIT_FOOD

        if pos in world.water:
            world.water.remove(pos)
            self.energy = min(MAX_ENERGY, self.energy + WATER_ENERGY_GAIN)
            self.water_drank += 1
            self.fitness += FIT_WATER

        if self.energy <= 0:
            self.alive = False

    def run_episode(self, world):
        self.reset(world)
        for _ in range(STEPS_PER_EPISODE):
            if not self.alive:
                break
            self.step(world)
            world.maybe_respawn()
        return EpisodeResult(
            fitness=self.fitness,
            lifespan=self.lifespan,
            food_eaten=self.food_eaten,
            water_drank=self.water_drank,
            explored_cells=len(self.explored),
            final_energy=self.energy,
        )


class GeneticAlgorithm:
    def __init__(self):
        self.population = [NetworkGenome() for _ in range(POP_SIZE)]
        self.generation = 0
        self.best_genome = None
        self.best_score = -1e9
        self.history_best = []
        self.last_stats = {}

    def tournament_pick(self, scored):
        picks = random.sample(scored, TOURNAMENT_SIZE)
        picks.sort(key=lambda x: x[0], reverse=True)
        return picks[0][1]

    def crossover(self, g1, g2):
        if random.random() > CROSSOVER_RATE:
            return g1.copy()
        size = len(g1.genes)
        p1 = random.randint(0, size - 1)
        p2 = random.randint(p1, size - 1)
        child_genes = g1.genes[:p1] + g2.genes[p1:p2] + g1.genes[p2:]
        return NetworkGenome(child_genes)

    def evaluate_population(self):
        scored = []
        lifespans = []
        for genome in self.population:
            world = World(WORLD_WIDTH, WORLD_HEIGHT)
            agent = Agent(genome)
            result = agent.run_episode(world)
            scored.append((result.fitness, genome, result))
            lifespans.append(result.lifespan)

        scored.sort(key=lambda x: x[0], reverse=True)
        scores = [s[0] for s in scored]
        best_fit = scores[0]
        mean_fit = sum(scores) / len(scores)
        median_fit = statistics.median(scores)

        best_result = scored[0][2]
        self.best_genome = scored[0][1].copy()
        self.best_score = best_fit
        self.history_best.append(best_fit)

        self.last_stats = {
            "generation": self.generation,
            "best": best_fit,
            "mean": mean_fit,
            "median": median_fit,
            "best_lifespan": best_result.lifespan,
            "best_energy": best_result.final_energy,
            "best_food": best_result.food_eaten,
            "best_water": best_result.water_drank,
            "max_lifespan": max(lifespans),
        }

        print(
            f"Gen {self.generation:4d} | best={best_fit:8.2f} "
            f"mean={mean_fit:7.2f} median={median_fit:7.2f} "
            f"best_life={best_result.lifespan:4d} food={best_result.food_eaten:3d} "
            f"water={best_result.water_drank:3d}"
        )

        elites = [scored[i][1].copy() for i in range(ELITE_COUNT)]
        flat_scored = [(s[0], s[1]) for s in scored]

        new_pop = elites
        while len(new_pop) < POP_SIZE:
            p1 = self.tournament_pick(flat_scored)
            p2 = self.tournament_pick(flat_scored)
            child = self.crossover(p1, p2)
            child.mutate()
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Evolution Forest Survival")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 14)

        self.ga = GeneticAlgorithm()
        self.world = World(WORLD_WIDTH, WORLD_HEIGHT)
        self.demo_agent = Agent(NetworkGenome())

        self.paused = False
        self.speed_index = 1
        self.show_extra = False
        self.running = True

        self.episode_step = 0
        self.start_new_demo()

    def start_new_demo(self):
        if self.ga.best_genome is None:
            self.ga.best_genome = self.ga.population[0].copy()
        self.world.reset()
        self.demo_agent = Agent(self.ga.best_genome.copy())
        self.demo_agent.reset(self.world)
        self.episode_step = 0

    def reset_evolution(self):
        self.ga = GeneticAlgorithm()
        self.start_new_demo()

    def train_generation(self):
        self.ga.evaluate_population()
        self.start_new_demo()

    def update_sim(self):
        if self.paused:
            return

        steps = STEPS_SPEEDS[self.speed_index]
        for _ in range(steps):
            if self.episode_step >= STEPS_PER_EPISODE or not self.demo_agent.alive:
                self.train_generation()
                break
            self.demo_agent.step(self.world)
            self.world.maybe_respawn()
            self.episode_step += 1

    def draw_world(self):
        world_offset_y = UI_HEIGHT
        for y in range(WORLD_HEIGHT):
            py = world_offset_y + y * TILE_SIZE
            for x in range(WORLD_WIDTH):
                px = x * TILE_SIZE
                rect = pygame.Rect(px, py, TILE_SIZE, TILE_SIZE)

                if (x, y) in self.world.trees:
                    color = TREE_COLOR
                elif (x, y) in self.world.food:
                    color = FOOD_COLOR
                elif (x, y) in self.world.water:
                    color = WATER_COLOR
                else:
                    color = EMPTY_COLOR

                pygame.draw.rect(self.screen, color, rect)
                if TILE_SIZE >= 10:
                    pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        ax = self.demo_agent.x * TILE_SIZE + TILE_SIZE // 2
        ay = world_offset_y + self.demo_agent.y * TILE_SIZE + TILE_SIZE // 2
        pygame.draw.circle(self.screen, AGENT_COLOR, (ax, ay), max(2, TILE_SIZE // 2 - 1))

        if self.show_extra:
            for _ in range(12):
                rx, ry = self.world.random_free_cell()
                px = rx * TILE_SIZE + TILE_SIZE // 2
                py = world_offset_y + ry * TILE_SIZE + TILE_SIZE // 2
                pygame.draw.circle(self.screen, (180, 80, 180), (px, py), 2)

    def draw_graph(self, x, y, w, h):
        history = self.ga.history_best
        pygame.draw.rect(self.screen, (30, 30, 34), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, (70, 70, 75), (x, y, w, h), 1, border_radius=4)
        if len(history) < 2:
            return
        min_v = min(history)
        max_v = max(history)
        span = max(1e-6, max_v - min_v)

        pts = []
        for i, val in enumerate(history[-w:]):
            px = x + i
            norm = (val - min_v) / span
            py = y + h - int(norm * (h - 4)) - 2
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(self.screen, GRAPH_COLOR, False, pts, 2)

    def draw_ui(self):
        pygame.draw.rect(self.screen, UI_COLOR, (0, 0, SCREEN_WIDTH, UI_HEIGHT))
        st = self.ga.last_stats
        if st:
            line1 = (
                f"Gen: {st['generation']}  Best: {st['best']:.2f}  Mean: {st['mean']:.2f}  "
                f"Median: {st['median']:.2f}"
            )
            line2 = (
                f"Episode step: {self.episode_step}/{STEPS_PER_EPISODE}  "
                f"Best lifespan: {st['best_lifespan']}  Energy: {self.demo_agent.energy:.1f}"
            )
        else:
            line1 = "Gen: 0  Best: N/A  Mean: N/A  Median: N/A"
            line2 = f"Episode step: {self.episode_step}/{STEPS_PER_EPISODE}  Energy: {self.demo_agent.energy:.1f}"

        line3 = (
            f"Speed: {STEPS_SPEEDS[self.speed_index]}x  "
            f"[SPACE pause] [TAB speed] [R reset] [V extra-visual={self.show_extra}]"
        )

        self.screen.blit(self.font.render(line1, True, (230, 230, 230)), (10, 8))
        self.screen.blit(self.font.render(line2, True, (220, 220, 180)), (10, 34))
        self.screen.blit(self.small_font.render(line3, True, (180, 200, 220)), (10, 63))

        self.draw_graph(SCREEN_WIDTH - 260, 10, 245, 80)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_TAB:
                        self.speed_index = (self.speed_index + 1) % len(STEPS_SPEEDS)
                    elif event.key == pygame.K_r:
                        self.reset_evolution()
                    elif event.key == pygame.K_v:
                        self.show_extra = not self.show_extra

            self.update_sim()
            self.screen.fill(BG_COLOR)
            self.draw_world()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
