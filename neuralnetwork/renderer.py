
import pygame

class Renderer():
    def __init__(self, grid_size=(10,10), cell_size=25):
        self.cell_size = cell_size
        self.grid_size = grid_size
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.grid_size[0] * self.cell_size, self.grid_size[1] * self.cell_size)
        )
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()


    def render(self,snake,food):
        self.screen.fill((0, 0, 0))

        head = None
        tail = None
        body = []
        for i, segment in enumerate(snake):
            rect = pygame.Rect(
                segment[0] * self.cell_size,
                segment[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )

            # Determine segment types
            if i == 0:  # Head
                head = rect
            elif i == len(snake) - 1:  # Tail
                tail = rect
            else:
                body.append((rect.centerx,rect.centery))

        body.insert(0,(head.centerx,head.centery))
        body.append((tail.centerx,tail.centery))
        pygame.draw.lines(self.screen, (0, 255, 0), False, body, width=self.cell_size//2)

        self._draw_head(head, 0)
        self._draw_tail(tail, 0)
    
        if food:
            print(food)
            
            if isinstance(food, list) or isinstance(food, set):
                for f in food:
                    food_rect = pygame.Rect(
                        f[0] * self.cell_size,
                        f[1] * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
            else:
                food_rect = pygame.Rect(
                    food[0] * self.cell_size,
                    food[1] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, (255, 0, 0), food_rect)

        pygame.display.flip()
        self.clock.tick(120)

    def _draw_head(self, rect, index):
        pygame.draw.rect(self.screen, (0, 255, 0), rect)  # Bright green for head
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (rect.x + self.cell_size // 4, rect.y + self.cell_size // 4),
            self.cell_size // 10,
        )
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            (rect.x + 3 * self.cell_size // 4, rect.y + self.cell_size // 4),
            self.cell_size // 10,
        )
        tongue_rect = pygame.Rect(
            rect.centerx - self.cell_size // 10,
            rect.bottom - self.cell_size // 8,
            self.cell_size // 5,
            self.cell_size // 8,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), tongue_rect)

    def _draw_tail(self, rect, index):
        return
        pygame.draw.rect(self.screen, (200, 200, 0), rect)

    def _draw_body(self, rect, index):
        prev_segment = self.snake[index - 1]
        next_segment = self.snake[index + 1]
        pygame.draw.rect(self.screen, (0, 200, 0), rect)  # Base green body

        # Directions to previous and next segments
        prev_dir = (prev_segment[0] - self.snake[index][0], prev_segment[1] - self.snake[index][1])
        next_dir = (next_segment[0] - self.snake[index][0], next_segment[1] - self.snake[index][1])

        BODY_HIGHLIGHT_COLOR = (0, 0, 255)

        if prev_dir[0] == 0 and next_dir[0] == 0:  # Vertical segment
            ...#pygame.draw.rect(self.screen, BODY_HIGHLIGHT_COLOR, rect.inflate(0, -self.cell_size // 5))

        elif prev_dir[1] == 0 and next_dir[1] == 0:  # Horizontal segment
            ...#pygame.draw.rect(self.screen, BODY_HIGHLIGHT_COLOR, rect.inflate(-self.cell_size // 5, 0))

        else:  # Handle corners
            size = self.cell_size // 2

            p_x, p_y = prev_dir
            n_x, n_y = next_dir

            if p_x > 0 and n_y > 0 or n_x > 0 and p_y > 0:  # Top-left corner
                corner_rect = pygame.Rect(rect.right + size // 2, rect.top+ size // 2, size, size)
            elif p_x > 0 and n_y < 0 or n_x > 0 and p_y < 0:  # Bottom-left corner
                corner_rect = pygame.Rect(rect.left + size // 2, rect.bottom - size, size, size*2)
            elif p_x < 0 and n_y > 0 or n_x < 0 and p_y > 0:  # Top-right corner
                corner_rect = pygame.Rect(rect.right, rect.top, size, size)
                pygame.draw.rect(self.screen, BODY_HIGHLIGHT_COLOR, corner_rect)
            elif p_x < 0 and n_y < 0 or n_x < 0 and p_y < 0:  # Bottom-right corner
                corner_rect = pygame.Rect(rect.right - size, rect.bottom - size, size, size)

    def close(self):
        pygame.quit()


    def save(self, n, score):
        pygame.image.save( self.screen, f'images/{score}_{n}_.png' )