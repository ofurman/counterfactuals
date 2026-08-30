import brands from '../../../../research/brand-assets.json'

const imageById = {
  pwr: new URL('../../assets/brands/pwr-logo.png', import.meta.url).href,
  genwro: new URL('../../assets/brands/genwro-logo.png', import.meta.url).href,
  tooploox: new URL('../../assets/brands/tpx-logo.png', import.meta.url).href,
}

export function BrandStrip({ side }: { side: 'left' | 'right' }) {
  const assets = brands.assets.filter((brand) => side === 'left' ? brand.id === 'pwr' : brand.id !== 'pwr')
  return (
    <div className={`brand-strip brand-strip--${side}`} data-brand-side={side} aria-label={side === 'left' ? 'PWr logo' : 'genwro.AI and Tooploox logos'}>
      {assets.map((brand) => (
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
