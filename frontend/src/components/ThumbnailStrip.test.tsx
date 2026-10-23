import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThumbnailStrip } from "./ThumbnailStrip";

afterEach(() => {
  cleanup();
});

describe("ThumbnailStrip", () => {
  it("renders nothing without pages", () => {
    const { container } = render(
      <ThumbnailStrip apiBase="" pages={[]} selectedPageIndex={0} onSelectPage={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("lets users switch pages from thumbnails", () => {
    const onSelectPage = vi.fn();
    render(
      <ThumbnailStrip
        apiBase=""
        pages={[
          { page: 1, image_url: "/jobs/j/pages/1.jpg", thumbnail_url: "/jobs/j/pages/1.jpg?width=120" },
          { page: 2, image_url: "/jobs/j/pages/2.jpg" },
        ]}
        selectedPageIndex={0}
        onSelectPage={onSelectPage}
      />,
    );

    fireEvent.click(screen.getByLabelText("Page 2"));
    expect(onSelectPage).toHaveBeenCalledWith(1);
    expect(screen.getByLabelText("Page 1")).toHaveAttribute("aria-current", "page");
    const images = document.querySelectorAll("img");
    expect(images[0]).toHaveAttribute("src", "/jobs/j/pages/1.jpg?width=120");
    expect(images[1]).toHaveAttribute("src", "/jobs/j/pages/2.jpg?width=120");
  });
});
