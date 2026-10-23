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
          { page: 1, image_url: "/jobs/j/pages/1.jpg" },
          { page: 2, image_url: "/jobs/j/pages/2.jpg" },
        ]}
        selectedPageIndex={0}
        onSelectPage={onSelectPage}
      />,
    );

    fireEvent.click(screen.getByLabelText("Page 2"));
    expect(onSelectPage).toHaveBeenCalledWith(1);
    expect(screen.getByLabelText("Page 1")).toHaveAttribute("aria-current", "page");
  });
});
