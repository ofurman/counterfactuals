import brands from '../../../../research/brand-assets.json'

const imageById = {
  pwr: new URL('../../assets/brands/pwr-logo.png', import.meta.url).href,
  genwro: new URL('../../assets/brands/genwro-logo.png', import.meta.url).href,
  tooploox: new URL('../../assets/brands/tpx-logo.png', import.meta.url).href,
}

export function BrandStrip() {
  return (
    <div className="brand-strip" aria-label="PWr, genwro.AI, and Tooploox logos">
      {brands.assets.map((brand) => (
        <img
          key={brand.id}
          data-brand-id={brand.id}
          className={`brand-logo brand-logo--${brand.id}`}
          src={imageById[brand.id as keyof typeof imageById]}
          alt={brand.label}
        />
      ))}
    </div>
  )
}
