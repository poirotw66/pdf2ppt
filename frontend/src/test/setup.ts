import "@testing-library/jest-dom/vitest";

class ResizeObserverMock {
	observe(): void {}
	disconnect(): void {}
	unobserve(): void {}
}

Object.defineProperty(globalThis, "ResizeObserver", {
	writable: true,
	configurable: true,
	value: ResizeObserverMock,
});