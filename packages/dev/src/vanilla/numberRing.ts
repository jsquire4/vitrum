// numberRing.ts — fixed-size circular moving-average buffer.

export class NumberRing {
  #values: number[];
  #index = 0;
  #filled = 0;

  constructor(size: number) {
    this.#values = new Array<number>(Math.max(1, size)).fill(0);
  }

  push(value: number): void {
    this.#values[this.#index] = value;
    this.#index = (this.#index + 1) % this.#values.length;
    this.#filled = Math.min(this.#filled + 1, this.#values.length);
  }

  mean(): number {
    if (this.#filled === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.#filled; i++) sum += this.#values[i] ?? 0;
    return sum / this.#filled;
  }
}
