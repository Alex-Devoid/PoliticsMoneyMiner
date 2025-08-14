import { test, expect } from '@playwright/test'

test.use({ baseURL: 'http://localhost:3000' })

test('QC flow: approve first page', async ({ page }) => {
  // 1. overview
  await page.goto('/filings')
  await page.getByText('Jared Cerullo').click()

  // 2. dashboard loads
  await expect(page.locator('text=PDF QC Dashboard')).toBeVisible()

  // 3. click approve
  await page.getByRole('button', { name:/approve this page/i }).click()

  // 4. all rows should now be green
  await expect(page.locator('tr.bg-green-lighten-5')).toHaveCount(10)

  // 5. sidebar chip in overview flips to green (optional)
})
