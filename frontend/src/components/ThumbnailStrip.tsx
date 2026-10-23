type ThumbnailPage = {
  page: number;
  image_url: string;
  thumbnail_url?: string | null;
};

type ThumbnailStripProps = {
  apiBase: string;
  pages: ThumbnailPage[];
  selectedPageIndex: number;
  onSelectPage: (index: number) => void;
};

export function ThumbnailStrip({ apiBase, pages, selectedPageIndex, onSelectPage }: ThumbnailStripProps) {
  if (pages.length === 0) {
    return null;
  }

  return (
    <div className="thumbnail-strip" data-testid="thumbnail-strip" aria-label="Page thumbnails">
      {pages.map((page, index) => {
        const src = `${apiBase}${page.thumbnail_url || `${page.image_url}?width=120`}`;
        return (
          <button
            key={page.page}
            type="button"
            className={index === selectedPageIndex ? "thumbnail-chip active" : "thumbnail-chip"}
            onClick={() => onSelectPage(index)}
            aria-label={`Page ${page.page}`}
            aria-current={index === selectedPageIndex ? "page" : undefined}
          >
            <img src={src} alt="" draggable={false} width={120} loading="lazy" />
            <span>Page {page.page}</span>
          </button>
        );
      })}
    </div>
  );
}
