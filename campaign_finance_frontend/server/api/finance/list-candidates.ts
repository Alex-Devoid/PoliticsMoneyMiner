// server/api/finance/list-candidates.ts
import { callFastApi } from '../_utils'      // adjust the relative path if needed
import type { H3Event } from 'h3'

export default async function (event: H3Event) {
  // forward *all* query-string params (state, limit, …)
  const query = getQuery(event)
  return callFastApi(event, '/finance/list/candidates', query)
}
