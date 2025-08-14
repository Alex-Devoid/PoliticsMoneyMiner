import { defineEventHandler } from 'h3'
import { callFastApi }        from '../_utils'   

export default defineEventHandler(async (event) => {
  // no query-params needed
  return await callFastApi(event, '/finance/states')
})
