import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { render, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ProjectQr } from './components/poster/ProjectQr'
import { posterData } from './data/posterData'

describe('GitHub-branded project QR', () => {
  it('uses a caption-free accessible link and a centered logo with high error correction', () => {
    const { container } = render(<ProjectQr />)
    const link = within(container).getByRole('link', { name: 'Open the CEL project repository' })
    expect(link).toHaveAttribute('href', posterData.identity.links.repository)
    expect(link.textContent).toBe('')
    const qr = link.querySelector('svg')!
    expect(qr).toHaveAttribute('width', '96')
    expect(qr).toHaveAttribute('height', '96')
    expect(qr).toHaveAttribute('data-qr-error-level', 'H')
    expect(qr).toHaveAttribute('data-qr-margin', '4')
    const logo = qr.querySelector('image')!
    const source = logo.getAttribute('href')!
    expect(source).toMatch(/^data:image\/svg\+xml/)
    const payload = source.slice(source.indexOf(',') + 1)
    const xml = source.includes(';base64,') ? Buffer.from(payload, 'base64').toString() : decodeURIComponent(payload)
    const parser = new DOMParser()
    const original = fs.readFileSync(path.join(process.cwd(), 'src/assets/qr/GitHub_Invertocat_Black_Clearspace.svg'), 'utf8')
    expect(parser.parseFromString(xml, 'image/svg+xml').querySelector('path')?.getAttribute('d'))
      .toBe(parser.parseFromString(original, 'image/svg+xml').querySelector('path')?.getAttribute('d'))
    const box = qr.getAttribute('viewBox')!.split(' ').map(Number)
    expect(Number(logo.getAttribute('x')) + Number(logo.getAttribute('width')) / 2).toBeCloseTo(box[2] / 2)
    expect(Number(logo.getAttribute('y')) + Number(logo.getAttribute('height')) / 2).toBeCloseTo(box[3] / 2)
  })

  it('preserves the official GitHub SVG artwork and its built-in clear space', () => {
    const bytes = fs.readFileSync(path.join(process.cwd(), 'src/assets/qr/GitHub_Invertocat_Black_Clearspace.svg'))
    expect(createHash('sha256').update(bytes).digest('hex')).toBe('b5e2f4b3f953f075de7016faa952d6ce3ec7492aa7b32408505ec25857962d89')
    expect(bytes.toString()).toContain('viewBox="0 0 128 128"')
  })
})
